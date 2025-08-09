import os
os.environ['MINERU_MODEL_SOURCE'] = "local"
import sys
import json
import time
import traceback
import re

from mineru.backend.pipeline.pipeline_analyze import ModelSingleton

from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

# log 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from log.logger import Logger
log = Logger()


class PDFTool:
    def __init__(self, lang="ch"):
        self.lang = lang
        try:
            self.model_manager = ModelSingleton()
            self.model_manager.get_model(lang=self.lang, formula_enable=True, table_enable=True)
            log.info("模型初始化完成")
        except Exception as e:
            log.exception(e)
    

    def extract_pdf_mid_data_v2(self, middle_json, image_dir):
        pdf_datas = []
        pdf_image_datas = []
        pdf_table_datas = []

        pdf_info = middle_json.get('pdf_info', [])
        if not isinstance(pdf_info, list):
            pdf_info = [pdf_info]  # v2有时为单独dict，强制转list兼容

        try:
            for data_page in pdf_info:
                para_blocks_new = []
                para_blocks = data_page.get('para_blocks', [])
                tables = data_page.get('tables', [])
                for table_info in tables:
                    table_blocks = table_info.get('blocks', [])
                    if len(table_blocks) < 2:
                        continue
                    table_caption_content = ""
                    table_body_html = ""
                    table_body_image = ""
                    for table_block_temp in table_blocks:
                        if table_block_temp.get('type') == 'table_caption':
                            table_caption = table_block_temp
                            table_caption_line_list = []
                            for table_caption_line in table_caption.get('lines', []):
                                temp_lien_content = ''.join([x.get('content', "") for x in table_caption_line.get('spans', [])])
                                table_caption_line_list.append(temp_lien_content)
                            table_caption_content = ' '.join(table_caption_line_list)
                        if table_block_temp.get('type') == 'table_body':
                            body_lines = table_block_temp.get('lines', [])
                            if body_lines and body_lines[0]['spans']:
                                table_body = body_lines[0]['spans'][0]
                                table_body_html = table_body.get('html', '')
                                image_path = table_body.get('image_path', '')
                                table_body_image = os.path.join(image_dir, image_path) if image_path else ""
                    pdf_table_datas.append({
                        "caption": table_caption_content,
                        "html": table_body_html,
                        "path": table_body_image
                    })
                images = data_page.get('images', [])
                for image_info in images:
                    image_block = image_info.get('blocks', [])
                    if len(image_block) < 2:
                        continue
                    image_caption_content = ""
                    image_path = ""
                    for temp_image_block in image_block:
                        if temp_image_block.get('type') == 'image_body':
                            body_lines = temp_image_block.get('lines', [])
                            if body_lines and body_lines[0]['spans']:
                                image_path_val = body_lines[0]['spans'][0].get('image_path', '')
                                image_path = os.path.join(image_dir, image_path_val) if image_path_val else ""
                        if temp_image_block.get('type') == 'image_caption':
                            image_caption_line_list = []
                            for image_caption_line in temp_image_block.get('lines', []):
                                temp_lien_content = ' '.join([x.get('content', '') for x in image_caption_line.get('spans', [])])
                                image_caption_line_list.append(temp_lien_content)
                            image_caption_content = ' '.join(image_caption_line_list)
                    pdf_image_datas.append({
                        "caption": image_caption_content,
                        "path": image_path
                    })
                for data_block in para_blocks:
                    block_type = data_block.get('type')
                    if block_type not in ['text', 'title']:
                        continue
                    lines = data_block.get('lines', [])
                    merged_content = ""
                    block_bbox_list = []
                    for data_line in lines:
                        for data_span in data_line.get('spans', []):
                            merged_content += data_span.get("content", "") + " "
                        bbox = data_line.get('bbox')
                        if bbox:
                            block_bbox_list.append(self.bbox_format(bbox))
                    para_blocks_new.append({
                        "type": block_type,
                        "context": merged_content.replace("  ", " "),
                        "bbox": block_bbox_list
                    })
                pdf_datas.append({
                    "page_num": data_page.get('page_idx', 0),
                    "page_size": data_page.get('page_size'),
                    "para_blocks": para_blocks_new
                })
            return pdf_datas, pdf_image_datas, pdf_table_datas
        except Exception:
            print(traceback.format_exc())
            print('pdf_mid_v2转换异常')
            return "", [], []

    def bbox_format(self, bbox):
        x0, y0, x1, y1 = bbox
        x2, y2 = x1, y0
        x3, y3 = x0, y1
        return [x0, y0, x2, y2, x1, y1, x3, y3]

    def pdf2md_minerU_v2(
        self,
        file_path,
        pdf_id,
        pip_type="txt",
        output_root="./tmp_parse_md",
        lang=None,
        start_page_id=0,
        end_page_id=None
    ):
        try:
            lang = lang if lang else self.lang   # 优先外部参数，否则用自身
            pdf_bytes = read_fn(file_path)
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            file_name = str(pdf_id)
            pdf_bytes_list = [pdf_bytes]
            lang_list = [lang]

            parse_method = pip_type
            infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(
                pdf_bytes_list,
                lang_list,
                parse_method=parse_method,
                formula_enable=True,
                table_enable=True
            )
            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_doc = all_pdf_docs[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]

            local_image_dir, local_md_dir = prepare_env(output_root, file_name, parse_method)
            image_writer = FileBasedDataWriter(local_image_dir)

            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable, True
            )
            pdf_info = middle_json["pdf_info"]
            image_dir = str(os.path.basename(local_image_dir))
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            mid_data, pdf_image_datas, pdf_table_datas = self.extract_pdf_mid_data_v2(middle_json, image_dir)
            return md_content.strip(), mid_data, pdf_image_datas, pdf_table_datas
        except Exception:
            print(f'pdf转md异常， pdf id: {file_path}, 异常信息为：\n{traceback.format_exc()}')
            return None, None, [], []
    
    def pdf2text(self, file_path: str, remove_header_footer=True) -> str:
        import fitz
        text_content = ""
        try:
            def get_text(page, remove_header_footer):
                clip = None
                if remove_header_footer:
                    height = 50
                    rect = page.rect
                    clip = fitz.Rect(0, height, rect.width, rect.height - height)
                return page.get_text(clip=clip)

            with fitz.open(file_path) as doc:
                text_content_list = [get_text(page, remove_header_footer) for page in doc]
            text_content = chr(12).join(text_content_list)
            # 去除非段落换行符号
            text_content = re.sub(r"\u200b\n|-\n| \n|\t\n", "",
                                  text_content.replace("\t\n\u0007\n", "\t ").replace("\n ", " ").replace(" \n\t", "\n\t"))
            text_content = text_content.replace('\n', '\n\n')
            if not self.detect_malformed_characters(text_content):
                text_content = ""
                raise Exception("pdf解析内容乱码占比过多")
            if not text_content:
                log.info(f'pdf无法解析')
        except Exception:
            log.error(f'{file_path} pdf转text异常，, 异常信息为：{traceback.print_exc()}')
        finally:
            return text_content.strip()
    
    def detect_malformed_characters(self, text, threshold=0.2):
        malformed_pattern = r"(||||||||||||||||||)"
        matches = re.findall(malformed_pattern, text)
        if len(matches) and len(matches)/len(text) > threshold:
             return False
        else:
            return True

# 测试程序入口
if __name__ == '__main__':
    pdf_tool = PDFTool(lang="ch")
    test_pdf = "/home/ninemax/work/bmy/pdf_api_server_原版_7-25/pdf_2.0_test/测试pdf/基于GM(1,1)模型的新能源预测及分析_杨少梅.pdf"
    log.info("=" * 50)
    log.info("开始PDF解析测试...")
    md_content, mid_data, pdf_image_datas, pdf_table_datas = pdf_tool.pdf2md_minerU_v2(test_pdf, "test_pdf_id")
    
    if md_content is not None and mid_data is not None:
        with open("result.md", "w", encoding="utf-8") as f:
            f.write(md_content or "")
        with open("mid_data.json", "w", encoding="utf-8") as f:
            json.dump({
                "mid_data": mid_data,
                "image_datas": pdf_image_datas,
                "table_datas": pdf_table_datas
            }, f, ensure_ascii=False, indent=2)
        log.info("解析成功！结果已保存到 result.md 和 mid_data.json")
        print("PDF解析成功，结果已保存到 result.md 和 mid_data.json")
    else:
        log.error("PDF解析失败，请查看日志和错误信息！")
        print("PDF解析失败，请查看日志和错误信息！")
    log.info("测试结束")
    log.info("=" * 50)
