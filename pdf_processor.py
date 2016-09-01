import popplerqt4
import sys
import PyQt4
import json
from os import listdir
from os.path import isfile, join, split, abspath
import codecs
import subprocess
from os import chdir, rename, remove
import Queue
import threading
import ann_utils as util

# sapienta path
sapienta_path = r'/home/ubuntu/Documents/sapienta'
# max thread numbers for sapienta anntoation
thread_num_sapienta = 10
# max thread numbers for extracting highlights
thread_num_highlights = 10


# extract highlights from a PDF file
def extract_pdf_highlights(pdf_path, output_path):
        doc = popplerqt4.Poppler.Document.load(pdf_path)
        total_annotations = 0
        ht = {}
        for i in range(doc.numPages()):
            page = doc.page(i)
            annotations = page.annotations()
            (pwidth, pheight) = (page.pageSize().width(), page.pageSize().height())
            if len(annotations) > 0:
                for annotation in annotations:
                    if isinstance(annotation, popplerqt4.Poppler.Annotation):
                        total_annotations += 1
                        if(isinstance(annotation, popplerqt4.Poppler.HighlightAnnotation)):
                            quads = annotation.highlightQuads()
                            txt = ""
                            for quad in quads:
                                rect = (quad.points[0].x() * pwidth,
                                        quad.points[0].y() * pheight,
                                        quad.points[2].x() * pwidth,
                                        quad.points[2].y() * pheight)
                                bdy = PyQt4.QtCore.QRectF()
                                bdy.setCoords(*rect)
                                txt = txt + unicode(page.text(bdy)) + ' '
                            key = str(i+1)
                            if key in ht:
                                ht[key].append(txt)
                            else:
                                ht[key] = [txt]
        p, f = split(pdf_path)
        with codecs.open(join(output_path, f[:f.find('.')] + '_ht.json'),
                         'w', encoding='utf-8') as write_file:
            json.dump(ht, write_file)


def sapienta_annotate(pdf_path, output_path):
    chdir(sapienta_path)
    proc = subprocess.Popen(["pdfxconv", "-a", pdf_path], stdout=subprocess.PIPE)
    streamdata = proc.communicate()[0]
    rc = proc.returncode
    if rc != 0:
        print('failed to annotate {0} with Sapienta'.format(pdf_path))
    else:
        head, tail = split(pdf_path)
        fname = tail[:tail.rfind('.')]
        ann_file_name = fname + '_annotated.xml'
        rename(join(head, ann_file_name), join(output_path, ann_file_name))
        remove(join(head, fname + '.xml'))


def sapienta_process(pdf_dir_path, opath):
    util.multi_thread_process_files(pdf_dir_path, 'pdf', thread_num_sapienta, sapienta_annotate,
                                    proc_desc='annotated by Sapienta',
                                    args=[opath])


def extract_highlights_process(pdf_dir_path, opath):
    util.multi_thread_process_files(pdf_dir_path, 'pdf', thread_num_highlights, extract_pdf_highlights,
                                    proc_desc='extracted highlights',
                                    args=[opath])

if __name__ == "__main__":
    pdf_file_path = abspath(sys.argv[1])
    output_path = abspath(sys.argv[2])
    sapienta_process(pdf_file_path, output_path)
    extract_highlights_process(pdf_file_path, output_path)
