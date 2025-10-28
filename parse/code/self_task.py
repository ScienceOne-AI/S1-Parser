import time
import os
from abc import ABC, abstractmethod
from pdf2image import convert_from_path
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset, ImageDataset
from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm

# 定义任务指令
TASK_INSTRUCTIONS = {
    'text': 'Please output the text content from the image.',
    'formula': 'Please write out the expression of the formula in the image using LaTeX format.',
    'table': 'Please output the table in the image in LaTeX format.'
}


class DataWriter(ABC):
    @abstractmethod
    def write(self, path: str, data: bytes) -> None:
        """Write the data to the file.

        Args:
            path (str): the target file where to write
            data (bytes): the data want to write
        """
        pass

    def write_string(self, path: str, data: str) -> None:
        """Write the data to file, the data will be encoded to bytes.

        Args:
            path (str): the target file where to write
            data (str): the data want to write
        """

        def safe_encode(data: str, method: str):
            try:
                bit_data = data.encode(encoding=method, errors='replace')
                return bit_data, True
            except:  # noqa
                return None, False

        for method in ['utf-8', 'ascii']:
            bit_data, flag = safe_encode(data, method)
            if flag:
                self.write(path, bit_data)
                break


class FileBasedDataWriter(DataWriter):
    def __init__(self, parent_dir: str = '') -> None:
        """Initialized with parent_dir.

        Args:
            parent_dir (str, optional): the parent directory that may be used within methods. Defaults to ''.
        """
        self._parent_dir = parent_dir

    def write(self, path: str, data: bytes) -> None:
        """Write file with data.

        Args:
            path (str): the path of file, if the path is relative path, it will be joined with parent_dir.
            data (bytes): the data want to write
        """
        fn_path = path
        if not os.path.isabs(fn_path) and len(self._parent_dir) > 0:
            fn_path = os.path.join(self._parent_dir, path)

        if not os.path.exists(os.path.dirname(fn_path)) and os.path.dirname(fn_path) != "":
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)

        with open(fn_path, 'wb') as f:
            f.write(data)


def single_task_recognition(input_file, output_dir, OCR_model, task):
    """
    Single task recognition for specific content type

    Args:
        input_file: Input file path
        output_dir: Output directory
        TaichuOCR_model: Pre-initialized model instance
        task: Task type ('text', 'formula', 'table')
    """
    print(f"Starting single task recognition: {task}")
    print(f"Processing file: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])

    # Prepare output directory
    local_md_dir = os.path.join(output_dir, name_without_suff)
    os.makedirs(local_md_dir, exist_ok=True)

    print(f"Output dir: {local_md_dir}")
    md_writer = FileBasedDataWriter(local_md_dir)

    # Get task instruction
    instruction = TASK_INSTRUCTIONS.get(task, TASK_INSTRUCTIONS['text'])

    # Check file type and prepare images
    file_extension = input_file.split(".")[-1].lower()
    images = []

    if file_extension == 'pdf':
        print("⚠️  WARNING: PDF input detected for single task recognition.")
        print("⚠️  WARNING: Converting all PDF pages to images for processing.")
        print("⚠️  WARNING: This may take longer and use more resources than image input.")
        print("⚠️  WARNING: Consider using individual images for better performance.")

        try:
            # Convert PDF pages to PIL images directly
            print("Converting PDF pages to images...")
            images = convert_from_path(input_file, dpi=150)
            print(f"Converted {len(images)} pages to images")

        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")

    elif file_extension in ['jpg', 'jpeg', 'png']:
        # Load single image
        from PIL import Image
        images = [Image.open(input_file)]
    else:
        raise ValueError(f"Single task recognition supports PDF and image files, got: {file_extension}")

    # Start recognition
    print(f"Performing {task} recognition on {len(images)} image(s)...")
    start_time = time.time()

    try:
        # Prepare instructions for all images
        instructions = [instruction] * len(images)

        # Use chat model for single task recognition with PIL images directly
        # responses = OCR_model.chat_model.batch_inference(images, instructions)

        responses = OCR_model.chat_model.batch_inference(images, instructions)

        recognition_time = time.time() - start_time
        print(f"Recognition time: {recognition_time:.2f}s")

        # Combine results
        combined_result = responses[0]
        for i, response in enumerate(responses):
            if i > 0:
                combined_result = combined_result + "\n\n" + response

        # Save result
        result_filename = f"{name_without_suff}_{task}_result.md"
        md_writer.write(result_filename, combined_result.encode('utf-8'))

        print(f"Single task recognition completed!")
        print(f"Task: {task}")
        print(f"Processed {len(images)} image(s)")
        print(f"Result saved to: {os.path.join(local_md_dir, result_filename)}")

        # Clean up resources
        try:
            # Give some time for async tasks to complete
            time.sleep(0.5)

            # Close images if they were opened
            for img in images:
                if hasattr(img, 'close'):
                    img.close()

        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")

        return local_md_dir

    except Exception as e:
        raise RuntimeError(f"Single task recognition failed: {str(e)}")


def parse_pdf(input_file, output_dir, OCR_model):
    """
    Parse PDF file and save results

    Args:
        input_file: Input PDF file path
        output_dir: Output directory
        OCR_model: Pre-initialized model instance
    """
    print(f"Starting to parse file: {input_file}")

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Get filename
    name_without_suff = '.'.join(os.path.basename(input_file).split(".")[:-1])

    # Prepare output directory
    local_image_dir = os.path.join(output_dir, name_without_suff, "images")
    local_md_dir = os.path.join(output_dir, name_without_suff)
    image_dir = os.path.basename(local_image_dir)
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)

    print(f"Output dir: {local_md_dir}")
    image_writer = FileBasedDataWriter(local_image_dir)
    md_writer = FileBasedDataWriter(local_md_dir)

    # Read file content
    reader = FileBasedDataReader()
    file_bytes = reader.read(input_file)

    # Create dataset instance
    file_extension = input_file.split(".")[-1].lower()
    if file_extension == "pdf":
        ds = PymuDocDataset(file_bytes)
    else:
        ds = ImageDataset(file_bytes)

    # Start inference
    print("Performing document parsing...")
    start_time = time.time()

    infer_result = ds.apply(doc_analyze_llm, OCR_model=OCR_model)

    # Pipeline processing
    pipe_result = infer_result.pipe_ocr_mode(image_writer, OCR_model=OCR_model)

    parsing_time = time.time() - start_time
    print(f"Parsing time: {parsing_time:.2f}s")

    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    pipe_result.dump_middle_json(md_writer, f'{name_without_suff}_middle.json')

    print("Results saved to ", local_md_dir)
    return local_md_dir
