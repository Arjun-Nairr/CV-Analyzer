import fitz

def extraction_of_cv(filepath):
  doc=fitz.open(filepath)

  text=""

  for i in doc:
    text+=i.get_text()

  return text


cv=extraction_of_cv("samplecv.pdf")

print(cv)
