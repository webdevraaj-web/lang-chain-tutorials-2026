from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

markdown='''

# 🚀 Python AI Project Guide

## 📌 Introduction
Artificial Intelligence (AI) is rapidly evolving and is being used in many real-world applications such as chatbots, recommendation systems, and automation tools.

## 🧠 Why Learn Python for AI?
Python is widely used in AI because of its simplicity and powerful libraries like NumPy, pandas, and TensorFlow.

---

## 🛠️ Basic Python Example

```python
def greet(name):
    return f"Hello, {name}! Welcome to AI learning."

print(greet("Raaj"))
'''


code='''

class Student:
    def __init__(self, name, age, marks):
        self.name = name
        self.age = age
        self.marks = marks

    def get_average(self):
        return sum(self.marks) / len(self.marks)

    def display_info(self):
        print(f"Name: {self.name}, Age: {self.age}")
        print(f"Marks: {self.marks}")


class School:
    def __init__(self, school_name):
        self.school_name = school_name
        self.students = []

    def add_student(self, student):
        self.students.append(student)

    def show_all_students(self):
        for student in self.students:
            student.display_info()
            print("-" * 30)

    def get_topper(self):
        if not self.students:
            return None
        return max(self.students, key=lambda s: s.get_average())


# Usage
s1 = Student("Raaj", 22, [80, 85, 90])
s2 = Student("Aman", 21, [70, 75, 72])

school = School("ABC School")
school.add_student(s1)
school.add_student(s2)

school.show_all_students()

topper = school.get_topper()
print(f"Topper: {topper.name}")
'''


spillter=RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=100,
    chunk_overlap=0,
    
)

result=spillter.split_text(markdown)

print(len(result))
print(result)