from typing import TypedDict

class User(TypedDict):
    name:str
    age:int

new_user:User={'name':'raaj','age':'30'}

print(new_user)