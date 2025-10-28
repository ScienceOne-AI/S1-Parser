"""
Classify PDF as text version or scanned version based on results from meta_scan.
Definition criteria:
1. What PDFs are text PDFs - meeting any of the following conditions:
  1. Randomly sample N pages, if any page has more than 100 text characters
  2. As long as there exists a page with 0 images
2. What are scanned PDFs - meeting any of the following conditions:
  1. ~~80% of pages have the same maximum image size and area exceeds 0.6 of page area~~
  2. Most pages have equal text length.
"""
import json
import sys
from collections import Counter

import click
import numpy as np
from loguru import logger

from magic_pdf.libs.commons import mymax, get_top_percent_list
from magic_pdf.filter.pdf_meta_scan import scan_max_page, junk_limit_min

TEXT_LEN_THRESHOLD = 100
AVG_TEXT_LEN_THRESHOLD = 100
TEXT_LEN_SAMPLE_RATIO = 0.1  # Sample 0.1 of pages for text length statistics

# A solution for merging images, combining certain special scanned version split images into one complete image
def merge_images(image_list, page_width, page_height, max_offset=5, max_gap=2):
    # First remove all overlapping bbox image data through set
    image_list_result = []
    for page_images in image_list:
        page_result = []
        dedup = set()
        for img in page_images:
            x0, y0, x1, y1, img_bojid = img
            if (x0, y0, x1, y1) in dedup:  # Some duplicate bboxes may appear, no need to repeat, need to remove
                continue
            else:
                dedup.add((x0, y0, x1, y1))
                page_result.append([x0, y0, x1, y1, img_bojid])
        image_list_result.append(page_result)

    # Next, merge images on the same page that can be stitched together
    merged_images = []
    for page_images in image_list_result:
        if not page_images:
            continue

        # First sort images on the same page from top to bottom, left to right
        page_images.sort(key=lambda img: (img[1], img[0]))

        merged = [page_images[0]]

        for img in page_images[1:]:
            x0, y0, x1, y1, imgid = img

            last_img = merged[-1]
            last_x0, last_y0, last_x1, last_y1, last_imgid = last_img

            # A single image width or height covering more than 90% of page width/height is a prerequisite for stitching
            full_width = abs(x1 - x0) >= page_width * 0.9
            full_height = abs(y1 - y0) >= page_height * 0.9

            # If width meets standard, check if can stitch vertically
            if full_width:
                # Vertical stitching needs two prerequisites: left and right boundaries can't offset more than max_offset, first image's bottom boundary and second image's top boundary can't offset more than max_gap
                close1 = (last_x0 - max_offset) <= x0 <= (last_x0 + max_offset) and (last_x1 - max_offset) <= x1 <= (
                            last_x1 + max_offset) and (last_y1 - max_gap) <= y0 <= (last_y1 + max_gap)

            # If height meets standard, check if can stitch horizontally
            if full_height:
                # Horizontal stitching needs two prerequisites: top and bottom boundaries can't offset more than max_offset, first image's right boundary and second image's left boundary can't offset more than max_gap
                close2 = (last_y0 - max_offset) <= y0 <= (last_y0 + max_offset) and (last_y1 - max_offset) <= y1 <= (
                            last_y1 + max_offset) and (last_x1 - max_gap) <= x0 <= (last_x1 + max_gap)

            # Check if the image can be merged with the last image
            if (full_width and close1) or (full_height and close2):
                # Merge the image with the last image
                merged[-1] = [min(x0, last_x0), min(y0, last_y0),
                              max(x1, last_x1), max(y1, last_y1), imgid]
            else:
                # Add the image as a new image
                merged.append(img)

        merged_images.append(merged)

    return merged_images


def classify_by_area(total_page: int, page_width, page_height, img_sz_list, text_len_list: list):
    """
    Returns False if 80% of pages have the same maximum image size and area exceeds 0.6 of page area, otherwise returns True
    """
    # # Only one page without images means it's a text PDF. But also needs to meet one condition, that is, there can't be any text on the page. Some scanned PDFs have blank pages with neither images nor text.
    # if any([len(img_sz) == 0 for img_sz in img_sz_list]):  # Contains pages without images
    #     # Now find the index of these pages
    #     empty_page_index = [i for i, img_sz in enumerate(img_sz_list) if len(img_sz) == 0]
    #     # Then check if there is any text on these pages
    #     text_len_at_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in empty_page_index and text_len > 0]
    #     if len(text_len_at_page_idx) > TEXT_LEN_THRESHOLD:  # No images, but has text, indicating it might be a text version, if no text then can't be determined, left for next step, now requires the text on this page to exceed a certain threshold
    #         return True

    # Remove images that appear more than 10 times by objid, these are hidden transparent layers with same id
    # First count occurrences of each id
    objid_cnt = Counter([objid for page_img_sz in img_sz_list for _, _, _, _, objid in page_img_sz])
    # Then remove those appearing more than 10 times
    if total_page >= scan_max_page:  # New meta_scan only scans first scan_max_page pages, when page count > scan_max_page, treat total_page as scan_max_page
        total_page = scan_max_page

    repeat_threshold = 2  # Set bad_image threshold to 2
    # repeat_threshold = min(2, total_page)  # When total_page is 1, repeat_threshold is 1, will cause misjudgment making all img become bad_img
    bad_image_objid = set([objid for objid, cnt in objid_cnt.items() if cnt >= repeat_threshold])
    # bad_image_page_idx = [i for i, page_img_sz in enumerate(img_sz_list) if any([objid in bad_image_objid for _, _, _, _, objid in page_img_sz])]
    # text_len_at_bad_image_page_idx = [text_len for i, text_len in enumerate(text_len_list) if i in bad_image_page_idx and text_len > 0]

    # Special case: a text PDF covers each page with a huge transparent image, huge means image covers more than 90% of page area
    # fake_image_ids = [objid for objid in bad_image_objid if
    #                   any([abs((x1 - x0) * (y1 - y0) / page_width * page_height) > 0.9 for images in img_sz_list for
    #                        x0, y0, x1, y1, _ in images])]  # Original code, any inside always true, reason？？？
    # fake_image_ids = [objid for objid in bad_image_objid for images in img_sz_list for x0, y0, x1, y1, img_id in images
    #                   if img_id == objid and abs((x1 - x0) * (y1 - y0)) / (page_width * page_height) > 0.9]

    # if len(fake_image_ids) > 0 and any([l > TEXT_LEN_THRESHOLD for l in text_len_at_bad_image_page_idx]):  # These transparent images' pages have text greater than threshold
    #     return True

    img_sz_list = [[img_sz for img_sz in page_img_sz if img_sz[-1] not in bad_image_objid] for page_img_sz in
                   img_sz_list]  # Filter out repeatedly appearing images

    # Some scanned versions split one page image into many, need to stitch images first then calculate
    img_sz_list = merge_images(img_sz_list, page_width, page_height)

    # Calculate maximum image area per page, then calculate ratio of this area to page area
    max_image_area_per_page = [mymax([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1, _ in page_img_sz]) for page_img_sz in
                               img_sz_list]
    page_area = page_width * page_height
    max_image_area_per_page = [area / page_area for area in max_image_area_per_page]
    max_image_area_per_page = [area for area in max_image_area_per_page if area > 0.5]

    if len(max_image_area_per_page) >= 0.5 * total_page:  # Threshold changed from 0.8 to 0.5, adapt to cases like 2 out of 3 pages and 1 out of 2 pages
        # Prerequisite for this condition is removing repeatedly appearing images. These are hidden transparent layers with same id
        return False
    else:
        return True


def classify_by_text_len(text_len_list: list, total_page: int):
    """
    Randomly sample 10% of pages, if less than 5 pages, take all pages.
    Check text length on pages, if any page has text length > TEXT_LEN_THRESHOLD, then it's text PDF
    """
    select_page_cnt = int(total_page * TEXT_LEN_SAMPLE_RATIO)  # Select 10% of pages
    if select_page_cnt < 5:
        select_page_cnt = total_page

    # # Excluding first and last 10 pages
    # if total_page > 20:  # If total pages > 20
    #     page_range = list(range(10, total_page - 10))  # From 11th page to the last 11th page
    # else:
    #     page_range = list(range(total_page))  # Otherwise select all pages
    # page_num = np.random.choice(page_range, min(select_page_cnt, len(page_range)), replace=False)
    # Excluding first and last 10 pages is awkward for PDFs with only 21, 22 pages, if the selected middle 1-2 pages happen to have no text, easy to misjudge, with avg_words rule, this rule can be ignored
    page_num = np.random.choice(total_page, select_page_cnt, replace=False)
    text_len_lst = [text_len_list[i] for i in page_num]
    is_text_pdf = any([text_len > TEXT_LEN_THRESHOLD for text_len in text_len_lst])
    return is_text_pdf


def classify_by_avg_words(text_len_list: list):
    """
    Supplementary rule: if average words per page < AVG_TEXT_LEN_THRESHOLD, not text PDF
    Mainly for various image collections
    """
    sum_words = sum(text_len_list)
    count_of_numbers = len(text_len_list)
    if count_of_numbers == 0:
        is_text_pdf = False
    else:
        avg_words = round(sum_words / count_of_numbers)
        if avg_words > AVG_TEXT_LEN_THRESHOLD:
            is_text_pdf = True
        else:
            is_text_pdf = False

    return is_text_pdf


def classify_by_img_num(img_sz_list: list, img_num_list: list):
    """
    Supplementary rule: there's a type of scanned PDF that puts all scanned pages on each page, which gets deduplicated during metascan,
    characteristic of this PDF's metascan result is all empty elements in img_sz_list, each page in img_num_list has large and same count
    """
    # Calculate number of non-empty elements in img_sz_list
    count_img_sz_list_not_none = sum(1 for item in img_sz_list if item)
    # Get top 80% elements
    top_eighty_percent = get_top_percent_list(img_num_list, 0.8)
    # Non-empty elements in img_sz_list <= 1, top 80% elements are all equal, and max value >= junk_limit_min
    if count_img_sz_list_not_none <= 1 and len(set(top_eighty_percent)) == 1 and max(img_num_list) >= junk_limit_min:
        return False  # If meets this condition, definitely not text PDF
    else:
        return True  # If doesn't meet these three conditions, might be text PDF, judge by other rules


def classify_by_text_layout(text_layout_per_page: list):
    """
    Judge if text layout is mainly vertical.

    Args:
        text_layout_per_page (list): Text layout list, each element represents text layout of one page,
                                     'vertical' means vertical layout, 'horizontal' means horizontal layout.

    Returns:
        bool: If text layout is mainly vertical, return False; otherwise return True.
    """
    # Count vertical layouts in text_layout_per_page
    count_vertical = sum(1 for item in text_layout_per_page if item == 'vertical')
    # Count horizontal layouts in text_layout_per_page
    count_horizontal = sum(1 for item in text_layout_per_page if item == 'horizontal')
    # Calculate ratio of vertical layouts in text_layout_per_page
    known_layout_cnt = count_vertical + count_horizontal
    if known_layout_cnt != 0:
        ratio = count_vertical / known_layout_cnt
        if ratio >= 0.5:  # Threshold set to 0.5, adapt to cases like 2 out of 3 pages and 1 out of 2 pages
            return False  # Text layout is mainly vertical, consider not text PDF
        else:
            return True  # Text layout is mainly horizontal, consider text PDF
    else:
        return False  # Text layout unknown, default consider not text PDF


def classify_by_img_narrow_strips(page_width, page_height, img_sz_list):
    """
    Judge if a page consists of narrow strips, two conditions:
    1. Image width or height reaches 90% of page width or height, and long side needs to be multiple times longer than short side
    2. 80% or more of all images on the entire page meet condition 1

    Args:
        page_width (float): Page width
        page_height (float): Page height
        img_sz_list (list): Image size list, each element is a tuple representing image rectangle area and size, format (x0, y0, x1, y1, size), where (x0, y0) is top-left corner coordinate, (x1, y1) is bottom-right corner coordinate, size is image size

    Returns:
        bool: If ratio of pages meeting conditions < 0.5, return True, otherwise return False
    """

    def is_narrow_strip(img):
        x0, y0, x1, y1, _ = img
        width, height = x1 - x0, y1 - y0
        return any([
            # Image width >= 90% of page width, and width >= 4 times height
            width >= page_width * 0.9 and width >= height * 4,
            # Image height >= 90% of page height, and height >= 4 times width
            height >= page_height * 0.9 and height >= width * 4,
        ])

    # Initialize count of pages meeting conditions
    narrow_strip_pages_count = 0

    # Traverse all pages
    for page_img_list in img_sz_list:
        # Ignore empty pages
        if not page_img_list:
            continue

        # Calculate total number of images on page
        total_images = len(page_img_list)

        # Calculate number of narrow strip images on page
        narrow_strip_images_count = 0
        for img in page_img_list:
            if is_narrow_strip(img):
                narrow_strip_images_count += 1
        # If narrow strip image count < 5, skip
        if narrow_strip_images_count < 5:
            continue
        else:
            # If narrow strip image ratio >= 0.8, increase count of pages meeting conditions
            if narrow_strip_images_count / total_images >= 0.8:
                narrow_strip_pages_count += 1

    # Calculate ratio of pages meeting conditions
    narrow_strip_pages_ratio = narrow_strip_pages_count / len(img_sz_list)

    return narrow_strip_pages_ratio < 0.5


def classify(total_page: int, page_width, page_height, img_sz_list: list, text_len_list: list, img_num_list: list,
             text_layout_list: list, invalid_chars: bool):
    """
    Image and page length units here are pts
    """
    results = {
        'by_image_area': classify_by_area(total_page, page_width, page_height, img_sz_list, text_len_list),
        'by_text_len': classify_by_text_len(text_len_list, total_page),
        'by_avg_words': classify_by_avg_words(text_len_list),
        'by_img_num': classify_by_img_num(img_sz_list, img_num_list),
        'by_text_layout': classify_by_text_layout(text_layout_list),
        'by_img_narrow_strips': classify_by_img_narrow_strips(page_width, page_height, img_sz_list),
        'by_invalid_chars': invalid_chars,
    }

    if all(results.values()):
        return True, results
    elif not any(results.values()):
        return False, results
    else:
        logger.warning(
            f"pdf is not classified by area and text_len, by_image_area: {results['by_image_area']},"
            f" by_text: {results['by_text_len']}, by_avg_words: {results['by_avg_words']}, by_img_num: {results['by_img_num']},"
            f" by_text_layout: {results['by_text_layout']}, by_img_narrow_strips: {results['by_img_narrow_strips']},"
            f" by_invalid_chars: {results['by_invalid_chars']}",
            file=sys.stderr)  # Use this situation to quickly find which PDFs are special, and fix classification algorithm accordingly
        return False, results


@click.command()
@click.option("--json-file", type=str, help="PDF information")
def main(json_file):
    if json_file is None:
        print("json_file is None", file=sys.stderr)
        exit(0)
    try:
        with open(json_file, "r") as f:
            for l in f:
                if l.strip() == "":
                    continue
                o = json.loads(l)
                total_page = o["total_page"]
                page_width = o["page_width_pts"]
                page_height = o["page_height_pts"]
                img_sz_list = o["image_info_per_page"]
                text_len_list = o['text_len_per_page']
                text_layout_list = o['text_layout_per_page']
                pdf_path = o['pdf_path']
                is_encrypted = o['is_encrypted']
                is_needs_password = o['is_needs_password']
                if is_encrypted or total_page == 0 or is_needs_password:  # Encrypted, password-required, no pages - don't process
                    continue
                tag = classify(total_page, page_width, page_height, img_sz_list, text_len_list, text_layout_list)
                o['is_text_pdf'] = tag
                print(json.dumps(o, ensure_ascii=False))
    except Exception as e:
        print("ERROR: ", e, file=sys.stderr)


if __name__ == "__main__":
    main()
    # false = False
    # true = True
    # null = None
    # o = {"pdf_path":"s3://llm-raw-snew/llm-raw-the-eye/raw/World%20Tracker%20Library/worldtracker.org/media/library/Science/Computer%20Science/Shreiner%20-%20OpenGL%20Programming%20Guide%206e%20%5BThe%20Redbook%5D%20%28AW%2C%202008%29.pdf","is_needs_password":false,"is_encrypted":false,"total_page":978,"page_width_pts":368,"page_height_pts":513,"image_info_per_page":[[[0,0,368,513,10037]],[[0,0,368,513,4]],[[0,0,368,513,7]],[[0,0,368,513,10]],[[0,0,368,513,13]],[[0,0,368,513,16]],[[0,0,368,513,19]],[[0,0,368,513,22]],[[0,0,368,513,25]],[[0,0,368,513,28]],[[0,0,368,513,31]],[[0,0,368,513,34]],[[0,0,368,513,37]],[[0,0,368,513,40]],[[0,0,368,513,43]],[[0,0,368,513,46]],[[0,0,368,513,49]],[[0,0,368,513,52]],[[0,0,368,513,55]],[[0,0,368,513,58]],[[0,0,368,513,61]],[[0,0,368,513,64]],[[0,0,368,513,67]],[[0,0,368,513,70]],[[0,0,368,513,73]],[[0,0,368,516,76]],[[0,0,368,516,79]],[[0,0,368,513,82]],[[0,0,368,513,85]],[[0,0,368,513,88]],[[0,0,368,513,91]],[[0,0,368,513,94]],[[0,0,368,513,97]],[[0,0,368,513,100]],[[0,0,368,513,103]],[[0,0,368,513,106]],[[0,0,368,513,109]],[[0,0,368,513,112]],[[0,0,368,513,115]],[[0,0,368,513,118]],[[0,0,368,513,121]],[[0,0,368,513,124]],[[0,0,368,513,127]],[[0,0,368,513,130]],[[0,0,368,513,133]],[[0,0,368,513,136]],[[0,0,368,513,139]],[[0,0,368,513,142]],[[0,0,368,513,145]],[[0,0,368,513,148]],[[0,0,368,513,151]],[[0,0,368,513,154]],[[0,0,368,513,157]],[[0,0,368,513,160]],[[0,0,368,513,163]],[[0,0,368,513,166]],[[0,0,368,513,169]],[[0,0,368,513,172]],[[0,0,368,513,175]],[[0,0,368,513,178]],[[0,0,368,513,181]],[[0,0,368,513,184]],[[0,0,368,513,187]],[[0,0,368,513,190]],[[0,0,368,513,193]],[[0,0,368,513,196]],[[0,0,368,513,199]],[[0,0,368,513,202]],[[0,0,368,513,205]],[[0,0,368,513,208]],[[0,0,368,513,211]],[[0,0,368,513,214]],[[0,0,368,513,217]],[[0,0,368,513,220]],[[0,0,368,513,223]],[[0,0,368,513,226]],[[0,0,368,513,229]],[[0,0,368,513,232]],[[0,0,368,513,235]],[[0,0,368,513,238]],[[0,0,368,513,241]],[[0,0,368,513,244]],[[0,0,368,513,247]],[[0,0,368,513,250]],[[0,0,368,513,253]],[[0,0,368,513,256]],[[0,0,368,513,259]],[[0,0,368,513,262]],[[0,0,368,513,265]],[[0,0,368,513,268]],[[0,0,368,513,271]],[[0,0,368,513,274]],[[0,0,368,513,277]],[[0,0,368,513,280]],[[0,0,368,513,283]],[[0,0,368,513,286]],[[0,0,368,513,289]],[[0,0,368,513,292]],[[0,0,368,513,295]],[[0,0,368,513,298]],[[0,0,368,513,301]],[[0,0,368,513,304]],[[0,0,368,513,307]],[[0,0,368,513,310]],[[0,0,368,513,313]],[[0,0,368,513,316]],[[0,0,368,513,319]],[[0,0,368,513,322]],[[0,0,368,513,325]],[[0,0,368,513,328]],[[0,0,368,513,331]],[[0,0,368,513,334]],[[0,0,368,513,337]],[[0,0,368,513,340]],[[0,0,368,513,343]],[[0,0,368,513,346]],[[0,0,368,513,349]],[[0,0,368,513,352]],[[0,0,368,513,355]],[[0,0,368,513,358]],[[0,0,368,513,361]],[[0,0,368,513,364]],[[0,0,368,513,367]],[[0,0,368,513,370]],[[0,0,368,513,373]],[[0,0,368,513,376]],[[0,0,368,513,379]],[[0,0,368,513,382]],[[0,0,368,513,385]],[[0,0,368,513,388]],[[0,0,368,513,391]],[[0,0,368,513,394]],[[0,0,368,513,397]],[[0,0,368,513,400]],[[0,0,368,513,403]],[[0,0,368,513,406]],[[0,0,368,513,409]],[[0,0,368,513,412]],[[0,0,368,513,415]],[[0,0,368,513,418]],[[0,0,368,513,421]],[[0,0,368,513,424]],[[0,0,368,513,427]],[[0,0,368,513,430]],[[0,0,368,513,433]],[[0,0,368,513,436]],[[0,0,368,513,439]],[[0,0,368,513,442]],[[0,0,368,513,445]],[[0,0,368,513,448]],[[0,0,368,513,451]],[[0,0,368,513,454]],[[0,0,368,513,457]],[[0,0,368,513,460]],[[0,0,368,513,463]],[[0,0,368,513,466]],[[0,0,368,513,469]],[[0,0,368,513,472]],[[0,0,368,513,475]],[[0,0,368,513,478]],[[0,0,368,513,481]],[[0,0,368,513,484]],[[0,0,368,513,487]],[[0,0,368,513,490]],[[0,0,368,513,493]],[[0,0,368,513,496]],[[0,0,368,513,499]],[[0,0,368,513,502]],[[0,0,368,513,505]],[[0,0,368,513,508]],[[0,0,368,513,511]],[[0,0,368,513,514]],[[0,0,368,513,517]],[[0,0,368,513,520]],[[0,0,368,513,523]],[[0,0,368,513,526]],[[0,0,368,513,529]],[[0,0,368,513,532]],[[0,0,368,513,535]],[[0,0,368,513,538]],[[0,0,368,513,541]],[[0,0,368,513,544]],[[0,0,368,513,547]],[[0,0,368,513,550]],[[0,0,368,513,553]],[[0,0,368,513,556]],[[0,0,368,513,559]],[[0,0,368,513,562]],[[0,0,368,513,565]],[[0,0,368,513,568]],[[0,0,368,513,571]],[[0,0,368,513,574]],[[0,0,368,513,577]],[[0,0,368,513,580]],[[0,0,368,513,583]],[[0,0,368,513,586]],[[0,0,368,513,589]],[[0,0,368,513,592]],[[0,0,368,513,595]],[[0,0,368,513,598]],[[0,0,368,513,601]],[[0,0,368,513,604]],[[0,0,368,513,607]],[[0,0,368,513,610]],[[0,0,368,513,613]],[[0,0,368,513,616]],[[0,0,368,513,619]],[[0,0,368,513,622]],[[0,0,368,513,625]],[[0,0,368,513,628]],[[0,0,368,513,631]],[[0,0,368,513,634]],[[0,0,368,513,637]],[[0,0,368,513,640]],[[0,0,368,513,643]],[[0,0,368,513,646]],[[0,0,368,513,649]],[[0,0,368,513,652]],[[0,0,368,513,655]],[[0,0,368,513,658]],[[0,0,368,513,661]],[[0,0,368,513,664]],[[0,0,368,513,667]],[[0,0,368,513,670]],[[0,0,368,513,673]],[[0,0,368,513,676]],[[0,0,368,513,679]],[[0,0,368,513,682]],[[0,0,368,513,685]],[[0,0,368,513,688]],[[0,0,368,513,691]],[[0,0,368,513,694]],[[0,0,368,513,697]],[[0,0,368,513,700]],[[0,0,368,513,703]],[[0,0,368,513,706]],[[0,0,368,513,709]],[[0,0,368,513,712]],[[0,0,368,513,715]],[[0,0,368,513,718]],[[0,0,368,513,721]],[[0,0,368,513,724]],[[0,0,368,513,727]],[[0,0,368,513,730]],[[0,0,368,513,733]],[[0,0,368,513,736]],[[0,0,368,513,739]],[[0,0,368,513,742]],[[0,0,368,513,745]],[[0,0,368,513,748]],[[0,0,368,513,751]],[[0,0,368,513,754]],[[0,0,368,513,757]],[[0,0,368,513,760]],[[0,0,368,513,763]],[[0,0,368,513,766]],[[0,0,368,513,769]],[[0,0,368,513,772]],[[0,0,368,513,775]],[[0,0,368,513,778]],[[0,0,368,513,781]],[[0,0,368,513,784]],[[0,0,368,513,787]],[[0,0,368,513,790]],[[0,0,368,513,793]],[[0,0,368,513,796]],[[0,0,368,513,799]],[[0,0,368,513,802]],[[0,0,368,513,805]],[[0,0,368,513,808]],[[0,0,368,513,811]],[[0,0,368,513,814]],[[0,0,368,513,817]],[[0,0,368,513,820]],[[0,0,368,513,823]],[[0,0,368,513,826]],[[0,0,368,513,829]],[[0,0,368,513,832]],[[0,0,368,513,835]],[[0,0,368,513,838]],[[0,0,368,513,841]],[[0,0,368,513,844]],[[0,0,368,513,847]],[[0,0,368,513,850]],[[0,0,368,513,853]],[[0,0,368,513,856]],[[0,0,368,513,859]],[[0,0,368,513,862]],[[0,0,368,513,865]],[[0,0,368,513,868]],[[0,0,368,513,871]],[[0,0,368,513,874]],[[0,0,368,513,877]],[[0,0,368,513,880]],[[0,0,368,513,883]],[[0,0,368,513,886]],[[0,0,368,513,889]],[[0,0,368,513,892]],[[0,0,368,513,895]],[[0,0,368,513,898]],[[0,0,368,513,901]],[[0,0,368,513,904]],[[0,0,368,513,907]],[[0,0,368,513,910]],[[0,0,368,513,913]],[[0,0,368,513,916]],[[0,0,368,513,919]],[[0,0,368,513,922]],[[0,0,368,513,925]],[[0,0,368,513,928]],[[0,0,368,513,931]],[[0,0,368,513,934]],[[0,0,368,513,937]],[[0,0,368,513,940]],[[0,0,368,513,943]],[[0,0,368,513,946]],[[0,0,368,513,949]],[[0,0,368,513,952]],[[0,0,368,513,955]],[[0,0,368,513,958]],[[0,0,368,513,961]],[[0,0,368,513,964]],[[0,0,368,513,967]],[[0,0,368,513,970]],[[0,0,368,513,973]],[[0,0,368,513,976]],[[0,0,368,513,979]],[[0,0,368,513,982]],[[0,0,368,513,985]],[[0,0,368,513,988]],[[0,0,368,513,991]],[[0,0,368,513,994]],[[0,0,368,513,997]],[[0,0,368,513,1000]],[[0,0,368,513,1003]],[[0,0,368,513,1006]],[[0,0,368,513,1009]],[[0,0,368,513,1012]],[[0,0,368,513,1015]],[[0,0,368,513,1018]],[[0,0,368,513,2797]],[[0,0,368,513,2798]],[[0,0,368,513,2799]],[[0,0,368,513,2800]],[[0,0,368,513,2801]],[[0,0,368,513,2802]],[[0,0,368,513,2803]],[[0,0,368,513,2804]],[[0,0,368,513,2805]],[[0,0,368,513,2806]],[[0,0,368,513,2807]],[[0,0,368,513,2808]],[[0,0,368,513,2809]],[[0,0,368,513,2810]],[[0,0,368,513,2811]],[[0,0,368,513,2812]],[[0,0,368,513,2813]],[[0,0,368,513,2814]],[[0,0,368,513,2815]],[[0,0,368,513,2816]],[[0,0,368,513,2817]],[[0,0,368,513,2818]],[[0,0,368,513,2819]],[[0,0,368,513,2820]],[[0,0,368,513,2821]],[[0,0,368,513,2822]],[[0,0,368,513,2823]],[[0,0,368,513,2824]],[[0,0,368,513,2825]],[[0,0,368,513,2826]],[[0,0,368,513,2827]],[[0,0,368,513,2828]],[[0,0,368,513,2829]],[[0,0,368,513,2830]],[[0,0,368,513,2831]],[[0,0,368,513,2832]],[[0,0,368,513,2833]],[[0,0,368,513,2834]],[[0,0,368,513,2835]],[[0,0,368,513,2836]],[[0,0,368,513,2837]],[[0,0,368,513,2838]],[[0,0,368,513,2839]],[[0,0,368,513,2840]],[[0,0,368,513,2841]],[[0,0,368,513,2842]],[[0,0,368,513,2843]],[[0,0,368,513,2844]],[[0,0,368,513,2845]],[[0,0,368,513,2846]],[[0,0,368,513,2847]],[[0,0,368,513,2848]],[[0,0,368,513,2849]],[[0,0,368,513,2850]],[[0,0,368,513,2851]],[[0,0,368,513,2852]],[[0,0,368,513,2853]],[[0,0,368,513,2854]],[[0,0,368,513,2855]],[[0,0,368,513,2856]],[[0,0,368,513,2857]],[[0,0,368,513,2858]],[[0,0,368,513,2859]],[[0,0,368,513,2860]],[[0,0,368,513,2861]],[[0,0,368,513,2862]],[[0,0,368,513,2863]],[[0,0,368,513,2864]],[[0,0,368,513,2797]],[[0,0,368,513,2798]],[[0,0,368,513,2799]],[[0,0,368,513,2800]],[[0,0,368,513,2801]],[[0,0,368,513,2802]],[[0,0,368,513,2803]],[[0,0,368,513,2804]],[[0,0,368,513,2805]],[[0,0,368,513,2806]],[[0,0,368,513,2807]],[[0,0,368,513,2808]],[[0,0,368,513,2809]],[[0,0,368,513,2810]],[[0,0,368,513,2811]],[[0,0,368,513,2812]],[[0,0,368,513,2813]],[[0,0,368,513,2814]],[[0,0,368,513,2815]],[[0,0,368,513,2816]],[[0,0,368,513,2817]],[[0,0,368,513,2818]],[[0,0,368,513,2819]],[[0,0,368,513,2820]],[[0,0,368,513,2821]],[[0,0,368,513,2822]],[[0,0,368,513,2823]],[[0,0,368,513,2824]],[[0,0,368,513,2825]],[[0,0,368,513,2826]],[[0,0,368,513,2827]],[[0,0,368,513,2828]],[[0,0,368,513,2829]],[[0,0,368,513,2830]],[[0,0,368,513,2831]],[[0,0,368,513,2832]],[[0,0,368,513,2833]],[[0,0,368,513,2834]],[[0,0,368,513,2835]],[[0,0,368,513,2836]],[[0,0,368,513,2837]],[[0,0,368,513,2838]],[[0,0,368,513,2839]],[[0,0,368,513,2840]],[[0,0,368,513,2841]],[[0,0,368,513,2842]],[[0,0,368,513,2843]],[[0,0,368,513,2844]],[[0,0,368,513,2845]],[[0,0,368,513,2846]],[[0,0,368,513,2847]],[[0,0,368,513,2848]],[[0,0,368,513,2849]],[[0,0,368,513,2850]],[[0,0,368,513,2851]],[[0,0,368,513,2852]],[[0,0,368,513,2853]],[[0,0,368,513,2854]],[[0,0,368,513,2855]],[[0,0,368,513,2856]],[[0,0,368,513,2857]],[[0,0,368,513,2858]],[[0,0,368,513,2859]],[[0,0,368,513,2860]],[[0,0,368,513,2861]],[[0,0,368,513,2862]],[[0,0,368,513,2863]],[[0,0,368,513,2864]],[[0,0,368,513,1293]],[[0,0,368,513,1296]],[[0,0,368,513,1299]],[[0,0,368,513,1302]],[[0,0,368,513,1305]],[[0,0,368,513,1308]],[[0,0,368,513,1311]],[[0,0,368,513,1314]],[[0,0,368,513,1317]],[[0,0,368,513,1320]],[[0,0,368,513,1323]],[[0,0,368,513,1326]],[[0,0,368,513,1329]],[[0,0,368,513,1332]],[[0,0,368,513,1335]],[[0,0,368,513,1338]],[[0,0,368,513,1341]],[[0,0,368,513,1344]],[[0,0,368,513,1347]],[[0,0,368,513,1350]],[[0,0,368,513,1353]],[[0,0,368,513,1356]],[[0,0,368,513,1359]],[[0,0,368,513,1362]],[[0,0,368,513,1365]],[[0,0,368,513,1368]],[[0,0,368,513,1371]],[[0,0,368,513,1374]],[[0,0,368,513,1377]],[[0,0,368,513,1380]],[[0,0,368,513,1383]],[[0,0,368,513,1386]],[[0,0,368,513,1389]],[[0,0,368,513,1392]],[[0,0,368,513,1395]],[[0,0,368,513,1398]],[[0,0,368,513,1401]],[[0,0,368,513,1404]],[[0,0,368,513,1407]],[[0,0,368,513,1410]],[[0,0,368,513,1413]],[[0,0,368,513,1416]],[[0,0,368,513,1419]],[[0,0,368,513,1422]],[[0,0,368,513,1425]],[[0,0,368,513,1428]],[[0,0,368,513,1431]],[[0,0,368,513,1434]],[[0,0,368,513,1437]],[[0,0,368,513,1440]],[[0,0,368,513,1443]],[[0,0,368,513,1446]],[[0,0,368,513,1449]],[[0,0,368,513,1452]],[[0,0,368,513,1455]],[[0,0,368,513,1458]],[[0,0,368,513,1461]],[[0,0,368,513,1464]],[[0,0,368,513,1467]],[[0,0,368,513,1470]],[[0,0,368,513,1473]],[[0,0,368,513,1476]],[[0,0,368,513,1479]],[[0,0,368,513,1482]],[[0,0,368,513,1485]],[[0,0,368,513,1488]],[[0,0,368,513,1491]],[[0,0,368,513,1494]],[[0,0,368,513,1497]],[[0,0,368,513,1500]],[[0,0,368,513,1503]],[[0,0,368,513,1506]],[[0,0,368,513,1509]],[[0,0,368,513,1512]],[[0,0,368,513,1515]],[[0,0,368,513,1518]],[[0,0,368,513,1521]],[[0,0,368,513,1524]],[[0,0,368,513,1527]],[[0,0,368,513,1530]],[[0,0,368,513,1533]],[[0,0,368,513,1536]],[[0,0,368,513,1539]],[[0,0,368,513,1542]],[[0,0,368,513,1545]],[[0,0,368,513,1548]],[[0,0,368,513,1551]],[[0,0,368,513,1554]],[[0,0,368,513,1557]],[[0,0,368,513,1560]],[[0,0,368,513,1563]],[[0,0,368,513,1566]],[[0,0,368,513,1569]],[[0,0,368,513,1572]],[[0,0,368,513,1575]],[[0,0,368,513,1578]],[[0,0,368,513,1581]],[[0,0,368,513,1584]],[[0,0,368,513,1587]],[[0,0,368,513,1590]],[[0,0,368,513,1593]],[[0,0,368,513,1596]],[[0,0,368,513,1599]],[[0,0,368,513,1602]],[[0,0,368,513,1605]],[[0,0,368,513,1608]],[[0,0,368,513,1611]],[[0,0,368,513,1614]],[[0,0,368,513,1617]],[[0,0,368,513,1620]],[[0,0,368,513,1623]],[[0,0,368,513,1626]],[[0,0,368,513,1629]],[[0,0,368,513,1632]],[[0,0,368,513,1635]],[[0,0,368,513,1638]],[[0,0,368,513,1641]],[[0,0,368,513,1644]],[[0,0,368,513,1647]],[[0,0,368,513,1650]],[[0,0,368,513,1653]],[[0,0,368,513,1656]],[[0,0,368,513,1659]],[[0,0,368,513,1662]],[[0,0,368,513,1665]],[[0,0,368,513,1668]],[[0,0,368,513,1671]],[[0,0,368,513,1674]],[[0,0,368,513,1677]],[[0,0,368,513,1680]],[[0,0,368,513,1683]],[[0,0,368,513,1686]],[[0,0,368,513,1689]],[[0,0,368,513,1692]],[[0,0,368,513,1695]],[[0,0,368,513,1698]],[[0,0,368,513,1701]],[[0,0,368,513,1704]],[[0,0,368,513,1707]],[[0,0,368,513,1710]],[[0,0,368,513,1713]],[[0,0,368,513,1716]],[[0,0,368,513,1719]],[[0,0,368,513,1722]],[[0,0,368,513,1725]],[[0,0,368,513,1728]],[[0,0,368,513,1731]],[[0,0,368,513,1734]],[[0,0,368,513,1737]],[[0,0,368,513,1740]],[[0,0,368,513,1743]],[[0,0,368,513,1746]],[[0,0,368,513,1749]],[[0,0,368,513,1752]],[[0,0,368,513,1755]],[[0,0,368,513,1758]],[[0,0,368,513,1761]],[[0,0,368,513,1764]],[[0,0,368,513,1767]],[[0,0,368,513,1770]],[[0,0,368,513,1773]],[[0,0,368,513,1776]],[[0,0,368,513,1779]],[[0,0,368,513,1782]],[[0,0,368,513,1785]],[[0,0,368,513,1788]],[[0,0,368,513,1791]],[[0,0,368,513,1794]],[[0,0,368,513,1797]],[[0,0,368,513,1800]],[[0,0,368,513,1803]],[[0,0,368,513,1806]],[[0,0,368,513,1809]],[[0,0,368,513,1812]],[[0,0,368,513,1815]],[[0,0,368,513,1818]],[[0,0,368,513,1821]],[[0,0,368,513,1824]],[[0,0,368,513,1827]],[[0,0,368,513,1830]],[[0,0,368,513,1833]],[[0,0,368,513,1836]],[[0,0,368,513,1839]],[[0,0,368,513,1842]],[[0,0,368,513,1845]],[[0,0,368,513,1848]],[[0,0,368,513,1851]],[[0,0,368,513,1854]],[[0,0,368,513,1857]],[[0,0,368,513,1860]],[[0,0,368,513,1863]],[[0,0,368,513,1866]],[[0,0,368,513,1869]],[[0,0,368,513,1872]],[[0,0,368,513,1875]],[[0,0,368,513,1878]],[[0,0,368,513,1881]],[[0,0,368,513,1884]],[[0,0,368,513,1887]],[[0,0,368,513,1890]],[[0,0,368,513,1893]],[[0,0,368,513,1896]],[[0,0,368,513,1899]],[[0,0,368,513,1902]],[[0,0,368,513,1905]],[[0,0,368,513,1908]],[[0,0,368,513,1911]],[[0,0,368,513,1914]],[[0,0,368,513,1917]],[[0,0,368,513,1920]],[[0,0,368,513,1923]],[[0,0,368,513,1926]],[[0,0,368,513,1929]],[[0,0,368,513,1932]],[[0,0,368,513,1935]],[[0,0,368,513,1938]],[[0,0,368,513,1941]],[[0,0,368,513,1944]],[[0,0,368,513,1947]],[[0,0,368,513,1950]],[[0,0,368,513,1953]],[[0,0,368,513,1956]],[[0,0,368,513,1959]],[[0,0,368,513,1962]],[[0,0,368,513,1965]],[[0,0,368,513,1968]],[[0,0,368,513,1971]],[[0,0,368,513,1974]],[[0,0,368,513,1977]],[[0,0,368,513,1980]],[[0,0,368,513,1983]],[[0,0,368,513,1986]],[[0,0,368,513,1989]],[[0,0,368,513,1992]],[[0,0,368,513,1995]],[[0,0,368,513,1998]],[[0,0,368,513,2001]],[[0,0,368,513,2004]],[[0,0,368,513,2007]],[[0,0,368,513,2010]],[[0,0,368,513,2013]],[[0,0,368,513,2016]],[[0,0,368,513,2019]],[[0,0,368,513,2022]],[[0,0,368,513,2025]],[[0,0,368,513,2028]],[[0,0,368,513,2031]],[[0,0,368,513,2034]],[[0,0,368,513,2037]],[[0,0,368,513,2040]],[[0,0,368,513,2043]],[[0,0,368,513,2046]],[[0,0,368,513,2049]],[[0,0,368,513,2052]],[[0,0,368,513,2055]],[[0,0,368,513,2058]],[[0,0,368,513,2061]],[[0,0,368,513,2064]],[[0,0,368,513,2067]],[[0,0,368,513,2070]],[[0,0,368,513,2073]],[[0,0,368,513,2076]],[[0,0,368,513,2079]],[[0,0,368,513,2082]],[[0,0,368,513,2085]],[[0,0,368,513,2088]],[[0,0,368,513,2091]],[[0,0,368,513,2094]],[[0,0,368,513,2097]],[[0,0,368,513,2100]],[[0,0,368,513,2103]],[[0,0,368,513,2106]],[[0,0,368,513,2109]],[[0,0,368,513,2112]],[[0,0,368,513,2115]],[[0,0,368,513,2118]],[[0,0,368,513,2121]],[[0,0,368,513,2124]],[[0,0,368,513,2127]],[[0,0,368,513,2130]],[[0,0,368,513,2133]],[[0,0,368,513,2136]],[[0,0,368,513,2139]],[[0,0,368,513,2142]],[[0,0,368,513,2145]],[[0,0,368,513,2148]],[[0,0,368,513,2151]],[[0,0,368,513,2154]],[[0,0,368,513,2157]],[[0,0,368,513,2160]],[[0,0,368,513,2163]],[[0,0,368,513,2166]],[[0,0,368,513,2169]],[[0,0,368,513,2172]],[[0,0,368,513,2175]],[[0,0,368,513,2178]],[[0,0,368,513,2181]],[[0,0,368,513,2184]],[[0,0,368,513,2187]],[[0,0,368,513,2190]],[[0,0,368,513,2193]],[[0,0,368,513,2196]],[[0,0,368,513,2199]],[[0,0,368,513,2202]],[[0,0,368,513,2205]],[[0,0,368,513,2208]],[[0,0,368,513,2211]],[[0,0,368,513,2214]],[[0,0,368,513,2217]],[[0,0,368,513,2220]],[[0,0,368,513,2223]],[[0,0,368,513,2226]],[[0,0,368,513,2229]],[[0,0,368,513,2232]],[[0,0,368,513,2235]],[[0,0,368,513,2238]],[[0,0,368,513,2241]],[[0,0,368,513,2244]],[[0,0,368,513,2247]],[[0,0,368,513,2250]],[[0,0,368,513,2253]],[[0,0,368,513,2256]],[[0,0,368,513,2259]],[[0,0,368,513,2262]],[[0,0,368,513,2265]],[[0,0,368,513,2268]],[[0,0,368,513,2271]],[[0,0,368,513,2274]],[[0,0,368,513,2277]],[[0,0,368,513,2280]],[[0,0,368,513,2283]],[[0,0,368,513,2286]],[[0,0,368,513,2289]],[[0,0,368,513,2292]],[[0,0,368,513,2295]],[[0,0,368,513,2298]],[[0,0,368,513,2301]],[[0,0,368,513,2304]],[[0,0,368,513,2307]],[[0,0,368,513,2310]],[[0,0,368,513,2313]],[[0,0,368,513,2316]],[[0,0,368,513,2319]],[[0,0,368,513,2322]],[[0,0,368,513,2325]],[[0,0,368,513,2328]],[[0,0,368,513,2331]],[[0,0,368,513,2334]],[[0,0,368,513,2337]],[[0,0,368,513,2340]],[[0,0,368,513,2343]],[[0,0,368,513,2346]],[[0,0,368,513,2349]],[[0,0,368,513,2352]],[[0,0,368,513,2355]],[[0,0,368,513,2358]],[[0,0,368,513,2361]],[[0,0,368,513,2364]],[[0,0,368,513,2367]],[[0,0,368,513,2370]],[[0,0,368,513,2373]],[[0,0,368,513,2376]],[[0,0,368,513,2379]],[[0,0,368,513,2382]],[[0,0,368,513,2385]],[[0,0,368,513,2388]],[[0,0,368,513,2391]],[[0,0,368,513,2394]],[[0,0,368,513,2397]],[[0,0,368,513,2400]],[[0,0,368,513,2403]],[[0,0,368,513,2406]],[[0,0,368,513,2409]],[[0,0,368,513,2412]],[[0,0,368,513,2415]],[[0,0,368,513,2418]],[[0,0,368,513,2421]],[[0,0,368,513,2424]],[[0,0,368,513,2427]],[[0,0,368,513,2430]],[[0,0,368,513,2433]],[[0,0,368,513,2436]],[[0,0,368,513,2439]],[[0,0,368,513,2442]],[[0,0,368,513,2445]],[[0,0,368,513,2448]],[[0,0,368,513,2451]],[[0,0,368,513,2454]],[[0,0,368,513,2457]],[[0,0,368,513,2460]],[[0,0,368,513,2463]],[[0,0,368,513,2466]],[[0,0,368,513,2469]],[[0,0,368,513,2472]],[[0,0,368,513,2475]],[[0,0,368,513,2478]],[[0,0,368,513,2481]],[[0,0,368,513,2484]],[[0,0,368,513,2487]],[[0,0,368,513,2490]],[[0,0,368,513,2493]],[[0,0,368,513,2496]],[[0,0,368,513,2499]],[[0,0,368,513,2502]],[[0,0,368,513,2505]],[[0,0,368,513,2508]],[[0,0,368,513,2511]],[[0,0,368,513,2514]],[[0,0,368,513,2517]],[[0,0,368,513,2520]],[[0,0,368,513,2523]],[[0,0,368,513,2526]],[[0,0,368,513,2529]],[[0,0,368,513,2532]],[[0,0,368,513,2535]],[[0,0,368,513,2538]],[[0,0,368,513,2541]],[[0,0,368,513,2544]],[[0,0,368,513,2547]],[[0,0,368,513,2550]],[[0,0,368,513,2553]],[[0,0,368,513,2556]],[[0,0,368,513,2559]],[[0,0,368,513,2562]],[[0,0,368,513,2565]],[[0,0,368,513,2568]],[[0,0,368,513,2571]],[[0,0,368,513,2574]],[[0,0,368,513,2577]],[[0,0,368,513,2580]],[[0,0,368,513,2583]],[[0,0,368,513,2586]],[[0,0,368,513,2589]],[[0,0,368,513,2592]],[[0,0,368,513,2595]],[[0,0,368,513,2598]],[[0,0,368,513,2601]],[[0,0,368,513,2604]],[[0,0,368,513,2607]],[[0,0,368,513,2610]],[[0,0,368,513,2613]],[[0,0,368,513,2616]],[[0,0,368,513,2619]],[[0,0,368,513,2622]],[[0,0,368,513,2625]],[[0,0,368,513,2628]],[[0,0,368,513,2631]],[[0,0,368,513,2634]],[[0,0,368,513,2637]],[[0,0,368,513,2640]],[[0,0,368,513,2643]],[[0,0,368,513,2646]],[[0,0,368,513,2649]],[[0,0,368,513,2652]],[[0,0,368,513,2655]],[[0,0,368,513,2658]],[[0,0,368,513,2661]],[[0,0,368,513,2664]],[[0,0,368,513,2667]],[[0,0,368,513,2670]],[[0,0,368,513,2673]],[[0,0,368,513,2676]],[[0,0,368,513,2679]],[[0,0,368,513,2682]],[[0,0,368,513,2685]],[[0,0,368,513,2688]],[[0,0,368,513,2691]],[[0,0,368,513,2694]],[[0,0,368,513,2697]],[[0,0,368,513,2700]],[[0,0,368,513,2703]],[[0,0,368,513,2706]],[[0,0,368,513,2709]],[[0,0,368,513,2712]],[[0,0,368,513,2715]],[[0,0,368,513,2718]],[[0,0,368,513,2721]],[[0,0,368,513,2724]],[[0,0,368,513,2727]],[[0,0,368,513,2730]],[[0,0,368,513,2733]],[[0,0,368,513,2736]],[[0,0,368,513,2739]],[[0,0,368,513,2742]],[[0,0,368,513,2745]],[[0,0,368,513,2748]],[[0,0,368,513,2751]],[[0,0,368,513,2754]],[[0,0,368,513,2757]],[[0,0,368,513,2760]],[[0,0,368,513,2763]],[[0,0,368,513,2766]],[[0,0,368,513,2769]],[[0,0,368,513,2772]],[[0,0,368,513,2775]],[[0,0,368,513,2778]],[[0,0,368,513,2781]],[[0,0,368,513,2784]],[[0,0,368,513,2787]],[[0,0,368,513,2790]],[[0,0,368,513,2793]],[[0,0,368,513,2796]]],"text_len_per_page":[53,53,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54,54],"metadata":{"format":"PDF 1.6","title":"","author":"","subject":"","keywords":"","creator":"Adobe Acrobat 7.0","producer":"Adobe Acrobat 7.0 Image Conversion Plug-in","creationDate":"D:20080404141457+01'00'","modDate":"D:20080404144821+01'00'","trapped":"","encryption":null}}
    # o = json.loads(json.dumps(o))
    # total_page = o["total_page"]
    # page_width = o["page_width_pts"]
    # page_height = o["page_height_pts"]
    # img_sz_list = o["image_info_per_page"]
    # text_len_list = o['text_len_per_page']
    # pdf_path = o['pdf_path']
    # is_encrypted = o['is_encrypted']
    # is_needs_password = o['is_needs_password']
    # if is_encrypted or total_page == 0 or is_needs_password:  # 加密的，需要密码的，没有页面的，都不处理
    #     print("加密的")
    #     exit(0)
    # tag = classify(pdf_path, total_page, page_width, page_height, img_sz_list, text_len_list)
    # o['is_text_pdf'] = tag
    # print(json.dumps(o, ensure_ascii=False))
