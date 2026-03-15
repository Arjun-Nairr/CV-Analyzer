###
cv="I am python sql master who is currently doing c++"
###
skills=["python","sql","c++","machine learning","java"]


def extract_skill():
  cv_skills=[]

  for j in cv.split():
    if j in skills:
      cv_skills.append(j)

  return cv_skills


extract_skill()
