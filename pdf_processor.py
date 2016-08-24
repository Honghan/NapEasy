import popplerqt4
import sys
import PyQt4
import json
from os import listdir
from os.path import isfile, join
import codecs


def extract_pdf_hightlights(pdf_path):

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
    return ht



def process_pdf(path):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    for f in onlyfiles:
        if f.endswith('.pdf'):
            ht = extract_pdf_hightlights(join(path,  f))
            with codecs.open(join(path, f[:f.find('.')] + '_ht.json'), 'w', encoding='utf-8') as write_file:
                json.dump(ht, write_file)


if __name__ == "__main__":
    process_pdf(sys.argv[1])

