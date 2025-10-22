import cv2
import os
import time
import base64
import numpy as np
from PIL import Image
from string import punctuation, digits, ascii_letters
from open_image_models import LicensePlateDetector
from groq import Groq
from paddleocr import PaddleOCR
import io
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class LicensePlateRecognition:
    def __init__(self, api_key, input_folder, output_folder):
        self.lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-640-license-plate-end2end")
        self.client = Groq(api_key=api_key)
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def encode_image(numpy_array):
        """Encodes a NumPy array to a base64 string."""
        image = Image.fromarray(np.uint8(numpy_array))
        buffer = io.BytesIO()
        # Use JPEG for smaller payloads; callers expect a data URL for image
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def perform_ocr(self, image_array):
        """Performs OCR on an image array and extracts text."""
        if image_array is None:
            raise ValueError("Image is None")
        try:
            results = self.ocr.ocr(image_array, cls=False, rec=True)
        except Exception as e:
            logger.exception("PaddleOCR failed on image: %s", e)
            return ""

        detected_text = []

        # Normalize/parse PaddleOCR output robustly. Paddle returns nested lists
        # where each line is typically (box, (text, confidence)). We'll try
        # extracting any string tokens we find.
        try:
            if not results:
                return ""

            for block in results:
                # block may be a list of lines or a tuple
                if not block:
                    continue
                # If block is a list of lines
                if isinstance(block, list):
                    for line in block:
                        # line may be (box, (text, conf)) or similar
                        if not line:
                            continue
                        # Try several indexing patterns
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            candidate = line[1]
                            if isinstance(candidate, (list, tuple)) and candidate:
                                text = candidate[0]
                            else:
                                text = str(candidate)
                            detected_text.append(str(text))
                        else:
                            detected_text.append(str(line))
                else:
                    # Single-line result shape
                    if isinstance(block, (list, tuple)) and len(block) >= 2:
                        candidate = block[1]
                        if isinstance(candidate, (list, tuple)) and candidate:
                            text = candidate[0]
                        else:
                            text = str(candidate)
                        detected_text.append(str(text))
                    else:
                        detected_text.append(str(block))
        except Exception:
            logger.exception("Error parsing OCR results")

        return ''.join(detected_text)

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Check file extension for common image types."""
        image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        _, ext = os.path.splitext(filename.lower())
        return ext in image_ext

    @staticmethod
    def draw_number_plate(frame, text, x1, y1, font_scale=1, thickness=2):
        """Draws a number plate with text on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        padding_x, padding_y = 20, 15
        plate_width = text_width + 2 * padding_x
        plate_height = text_height + 2 * padding_y

        new_x1, new_y1 = x1, y1 - plate_height // 2 - 100
        new_x2, new_y2 = new_x1 + plate_width, new_y1 + plate_height

        text_x = new_x1 + padding_x
        text_y = new_y1 + padding_y + text_height

        plate_color, border_color = (255, 255, 255), (0, 0, 0)
        cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), border_color, thickness=-1)
        cv2.rectangle(frame, (new_x1 + 5, new_y1 + 5), (new_x2 - 5, new_y2 - 5), plate_color, thickness=-1)

        shadow_color, text_color = (0, 0, 0), (50, 50, 50)
        cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, thickness=thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness=thickness)

        return frame

    def groq_plate_recognizer(self, crop_image):
        """Recognizes the license plate number using Groq API."""
        base64_image = self.encode_image(crop_image)
        try:
            response = self.client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "I have provided a cropped image of a vehicle's license plate. Please extract and return only the license plate number from the image, without any additional text or explanation.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=256,
                top_p=1,
                stream=False,
                stop=None,
            )
        except Exception:
            logger.exception("Groq API call failed for an image")
            return ""

        # Safely parse the response
        try:
            choice = getattr(response, 'choices', None) or response.get('choices') if isinstance(response, dict) else None
            if not choice:
                # try older attribute access
                content = getattr(response.choices[0].message, 'content', '') if hasattr(response, 'choices') else ''
            else:
                # Expect choice to be a list-like
                if isinstance(choice, list) and choice:
                    content = getattr(choice[0].message, 'content', None) if hasattr(choice[0], 'message') else choice[0].get('message', {}).get('content')
                else:
                    content = ''
        except Exception:
            logger.exception("Failed to parse Groq response")
            return ""

        return content or ""

    def process_images(self):
        """Processes each image in the input folder."""
        start_time = time.time()
        processed = 0
        skipped = 0
        errors = 0

        # List and filter files to image-only for stable behavior
        try:
            filenames = [f for f in sorted(os.listdir(self.input_folder)) if self._is_image_file(f)]
        except FileNotFoundError:
            logger.error("Input folder does not exist: %s", self.input_folder)
            return

        logger.info("Starting processing %d image(s) from %s", len(filenames), self.input_folder)

        for image_name in filenames:
            image_start = time.time()
            input_path = os.path.join(self.input_folder, image_name)
            output_path = os.path.join(self.output_folder, image_name)

            logger.info("Processing image: %s", image_name)

            try:
                frame = cv2.imread(input_path)
                if frame is None:
                    logger.warning("Skipping %s: Unable to read image", image_name)
                    skipped += 1
                    continue

                results = self.lp_detector.predict(frame)
                if results:
                    logger.debug("Detected %d candidate plate(s) in %s", len(results), image_name)
                    for box in results:
                        try:
                            x1, y1, x2, y2 = int(box.bounding_box.x1), int(box.bounding_box.y1), int(box.bounding_box.x2), int(box.bounding_box.y2)
                        except Exception:
                            logger.exception("Invalid bounding box for %s", image_name)
                            continue

                        # Guard crop coordinates
                        y1c, y2c = max(0, y1), min(frame.shape[0], y2)
                        x1c, x2c = max(0, x1), min(frame.shape[1], x2)
                        crop_image = frame[y1c:y2c, x1c:x2c]

                        ocr_text = self.perform_ocr(crop_image).translate(str.maketrans('', '', punctuation)).strip()

                        if (len(ocr_text) > 6 and len(ocr_text) < 32 and
                                ocr_text[0] in ascii_letters and any(ch in digits for ch in ocr_text)):
                            display_text = ocr_text
                        else:
                            display_text = self.groq_plate_recognizer(crop_image)

                        if display_text:
                            frame = self.draw_number_plate(frame, display_text, x1c, y1c)
                        else:
                            logger.debug("No plate text found for box in %s", image_name)

                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
                else:
                    logger.info("No number plate detected in %s", image_name)

                cv2.imwrite(output_path, frame)
                processed += 1
                logger.info("Finished %s in %.2fs", image_name, time.time() - image_start)

            except Exception:
                errors += 1
                logger.exception("Failed to process %s", image_name)

        total_time = time.time() - start_time
        logger.info("Processing completed: processed=%d skipped=%d errors=%d total_time=%.2fs", processed, skipped, errors, total_time)

# Usage example
if __name__ == "__main__":
    # Read configuration from environment variables with sensible defaults
    api_key = os.getenv("GROQ_API_KEY")
    input_folder = os.getenv("LPR_INPUT_FOLDER", "./collection")
    output_folder = os.getenv("LPR_OUTPUT_FOLDER", "./output_images")

    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.\n"
              "Set it in your environment or create a .env file with GROQ_API_KEY=<your_key>.")
        sys.exit(1)

    lpr = LicensePlateRecognition(api_key, input_folder, output_folder)
    lpr.process_images()import cv2
import os
import time
import base64
import numpy as np
from PIL import Image
from string import punctuation, digits, ascii_letters
from open_image_models import LicensePlateDetector
from groq import Groq
from paddleocr import PaddleOCR
import io
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

class LicensePlateRecognition:
    def __init__(self, api_key, input_folder, output_folder):
        self.lp_detector = LicensePlateDetector(detection_model="yolo-v9-t-640-license-plate-end2end")
        self.client = Groq(api_key=api_key)
        self.ocr = PaddleOCR(lang="en", use_angle_cls=True, show_log=False)
        self.input_folder = input_folder
        self.output_folder = output_folder

        # Create the output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def encode_image(numpy_array):
        """Encodes a NumPy array to a base64 string."""
        image = Image.fromarray(np.uint8(numpy_array))
        buffer = io.BytesIO()
        # Use JPEG for smaller payloads; callers expect a data URL for image
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def perform_ocr(self, image_array):
        """Performs OCR on an image array and extracts text."""
        if image_array is None:
            raise ValueError("Image is None")
        try:
            results = self.ocr.ocr(image_array, cls=False, rec=True)
        except Exception as e:
            logger.exception("PaddleOCR failed on image: %s", e)
            return ""

        detected_text = []

        # Normalize/parse PaddleOCR output robustly. Paddle returns nested lists
        # where each line is typically (box, (text, confidence)). We'll try
        # extracting any string tokens we find.
        try:
            if not results:
                return ""

            for block in results:
                # block may be a list of lines or a tuple
                if not block:
                    continue
                # If block is a list of lines
                if isinstance(block, list):
                    for line in block:
                        # line may be (box, (text, conf)) or similar
                        if not line:
                            continue
                        # Try several indexing patterns
                        if isinstance(line, (list, tuple)) and len(line) >= 2:
                            candidate = line[1]
                            if isinstance(candidate, (list, tuple)) and candidate:
                                text = candidate[0]
                            else:
                                text = str(candidate)
                            detected_text.append(str(text))
                        else:
                            detected_text.append(str(line))
                else:
                    # Single-line result shape
                    if isinstance(block, (list, tuple)) and len(block) >= 2:
                        candidate = block[1]
                        if isinstance(candidate, (list, tuple)) and candidate:
                            text = candidate[0]
                        else:
                            text = str(candidate)
                        detected_text.append(str(text))
                    else:
                        detected_text.append(str(block))
        except Exception:
            logger.exception("Error parsing OCR results")

        return ''.join(detected_text)

    @staticmethod
    def _is_image_file(filename: str) -> bool:
        """Check file extension for common image types."""
        image_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        _, ext = os.path.splitext(filename.lower())
        return ext in image_ext

    @staticmethod
    def draw_number_plate(frame, text, x1, y1, font_scale=1, thickness=2):
        """Draws a number plate with text on the image."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size

        padding_x, padding_y = 20, 15
        plate_width = text_width + 2 * padding_x
        plate_height = text_height + 2 * padding_y

        new_x1, new_y1 = x1, y1 - plate_height // 2 - 100
        new_x2, new_y2 = new_x1 + plate_width, new_y1 + plate_height

        text_x = new_x1 + padding_x
        text_y = new_y1 + padding_y + text_height

        plate_color, border_color = (255, 255, 255), (0, 0, 0)
        cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), border_color, thickness=-1)
        cv2.rectangle(frame, (new_x1 + 5, new_y1 + 5), (new_x2 - 5, new_y2 - 5), plate_color, thickness=-1)

        shadow_color, text_color = (0, 0, 0), (50, 50, 50)
        cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, shadow_color, thickness=thickness + 2)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, text_color, thickness=thickness)

        return frame

    def groq_plate_recognizer(self, crop_image):
        """Recognizes the license plate number using Groq API."""
        base64_image = self.encode_image(crop_image)
        try:
            response = self.client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "I have provided a cropped image of a vehicle's license plate. Please extract and return only the license plate number from the image, without any additional text or explanation.",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": base64_image},
                            },
                        ],
                    }
                ],
                temperature=0,
                max_tokens=256,
                top_p=1,
                stream=False,
                stop=None,
            )
        except Exception:
            logger.exception("Groq API call failed for an image")
            return ""

        # Safely parse the response
        try:
            choice = getattr(response, 'choices', None) or response.get('choices') if isinstance(response, dict) else None
            if not choice:
                # try older attribute access
                content = getattr(response.choices[0].message, 'content', '') if hasattr(response, 'choices') else ''
            else:
                # Expect choice to be a list-like
                if isinstance(choice, list) and choice:
                    content = getattr(choice[0].message, 'content', None) if hasattr(choice[0], 'message') else choice[0].get('message', {}).get('content')
                else:
                    content = ''
        except Exception:
            logger.exception("Failed to parse Groq response")
            return ""

        return content or ""

    def process_images(self):
        """Processes each image in the input folder."""
        start_time = time.time()
        processed = 0
        skipped = 0
        errors = 0

        # List and filter files to image-only for stable behavior
        try:
            filenames = [f for f in sorted(os.listdir(self.input_folder)) if self._is_image_file(f)]
        except FileNotFoundError:
            logger.error("Input folder does not exist: %s", self.input_folder)
            return

        logger.info("Starting processing %d image(s) from %s", len(filenames), self.input_folder)

        for image_name in filenames:
            image_start = time.time()
            input_path = os.path.join(self.input_folder, image_name)
            output_path = os.path.join(self.output_folder, image_name)

            logger.info("Processing image: %s", image_name)

            try:
                frame = cv2.imread(input_path)
                if frame is None:
                    logger.warning("Skipping %s: Unable to read image", image_name)
                    skipped += 1
                    continue

                results = self.lp_detector.predict(frame)
                if results:
                    logger.debug("Detected %d candidate plate(s) in %s", len(results), image_name)
                    for box in results:
                        try:
                            x1, y1, x2, y2 = int(box.bounding_box.x1), int(box.bounding_box.y1), int(box.bounding_box.x2), int(box.bounding_box.y2)
                        except Exception:
                            logger.exception("Invalid bounding box for %s", image_name)
                            continue

                        # Guard crop coordinates
                        y1c, y2c = max(0, y1), min(frame.shape[0], y2)
                        x1c, x2c = max(0, x1), min(frame.shape[1], x2)
                        crop_image = frame[y1c:y2c, x1c:x2c]

                        ocr_text = self.perform_ocr(crop_image).translate(str.maketrans('', '', punctuation)).strip()

                        if (len(ocr_text) > 6 and len(ocr_text) < 32 and
                                ocr_text[0] in ascii_letters and any(ch in digits for ch in ocr_text)):
                            display_text = ocr_text
                        else:
                            display_text = self.groq_plate_recognizer(crop_image)

                        if display_text:
                            frame = self.draw_number_plate(frame, display_text, x1c, y1c)
                        else:
                            logger.debug("No plate text found for box in %s", image_name)

                        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (255, 0, 0), 2)
                else:
                    logger.info("No number plate detected in %s", image_name)

                cv2.imwrite(output_path, frame)
                processed += 1
                logger.info("Finished %s in %.2fs", image_name, time.time() - image_start)

            except Exception:
                errors += 1
                logger.exception("Failed to process %s", image_name)

        total_time = time.time() - start_time
        logger.info("Processing completed: processed=%d skipped=%d errors=%d total_time=%.2fs", processed, skipped, errors, total_time)

# Usage example
if __name__ == "__main__":
    # Read configuration from environment variables with sensible defaults
    api_key = os.getenv("GROQ_API_KEY")
    input_folder = os.getenv("LPR_INPUT_FOLDER", "./collection")
    output_folder = os.getenv("LPR_OUTPUT_FOLDER", "./output_images")

    if not api_key:
        print("ERROR: GROQ_API_KEY environment variable is not set.\n"
              "Set it in your environment or create a .env file with GROQ_API_KEY=<your_key>.")
        sys.exit(1)

    lpr = LicensePlateRecognition(api_key, input_folder, output_folder)
    lpr.process_images()
