import fitz
doc = fitz.open(r'd:\SNU\Semester_6\motion_planning\project_retry\Robot soccer Striker.pdf')
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=200)
    out_path = f'd:\\SNU\\Semester_6\\motion_planning\\project_retry\\data\\pdf_inspection\\page_{i}.png'
    pix.save(out_path)
    print(f'Saved page {i} to {out_path} ({pix.width}x{pix.height})')
