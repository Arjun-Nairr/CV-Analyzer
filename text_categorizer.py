def extract_skill(cvtest):
  skills = ["python", "sql", "c\+\+", "machine learning", "java"] 
    
  pattern = r'\b(' + '|'.join(skills) + r')\b'
    
    
  matches = re.findall(pattern, cvtest.lower())

  return matches

import re

def extract_experience(exptest):
  
  matches = re.findall(r'\d+\s+years', exptest)

  return matches

def industry(industrytest):
    
    
    
    industries = {
        "Tech": r"\b(software|python|java|cloud|developer|data science|ai|ml)\b",
        "Finance": r"\b(banking|accounting|audit|fintech|investment|equity|excel)\b",
        "Healthcare": r"\b(medical|nurse|patient|clinical|hospital|pharmacy|surgery)\b"
    }
    
    for industry, pattern in industries.items():
        if re.search(pattern, industrytest):
            return industry
###testing

print(extract_skill(cv.lower()))
print(extract_experience(cv.lower()))
print(industry(cv.lower()))
