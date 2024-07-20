# class Person():
#     def setName(self, name:str):
#         self.name = name.upper()

#     def getName(self):
#         return self.name
    


# obj = Person()
# obj.setName('chris')
# print(obj.getName())



# obj_2 = Person()
# obj_2.setName('Noble')
# print(obj_2.getName())


# print(f"{obj.getName()}, and the other {obj_2.getName()}")

## CSC  ##
class Persona():
    def __init__(self, euler, pi, golden_ratio):
        self.euler, self.pi, self.golden_ratio = euler, pi, golden_ratio

    def add(self):
        return self.euler + self.pi * 12
    

val = Persona(1, 2, 3)
print(val.add())

print(Persona(2, 23, 45).euler)

##  CSC_JUPEB                              INHERITANCE                                 ##
class Data(Persona):
    def __init__(self, euler, pi, golden_ratio, dob, kin):

        super().__init__(euler, pi, golden_ratio)

        self.a = self.euler
    
a_1 = Data(1, 2, 3, 23, 'uche')
print(a_1.a)
print(a_1.add())

