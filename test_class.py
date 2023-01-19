# class Person:
#     def __call__(self, name):
#         print("call " + "Hello " + name)
    
#     def hello(self, name):
#         print("hello " + name)
    

# person = Person()
# person("zhangsan")
# person.hello("lisi")

 
class FooParent(object):
    def __init__(self):
        self.parent = 'I\'m the parent.'
        print ('Parent')
    
    def bar(self, message):
        print ("%s from Parent" % message)
 
class FooChild(FooParent):
    def __init__(self):
        # super(FooChild,self) 首先找到 FooChild 的父类（就是类 FooParent），然后把类 FooChild 的对象转换为类 FooParent 的对象
        super(FooChild, self).__init__()    
        print ('Child')
        
    def bar(self, message):
        super(FooChild, self).bar(message)
        print ('Child bar fuction')
        print (self.parent)     ## 有父类init里的attribute
 
if __name__ == '__main__':
    fooChild = FooChild()
    print("--------------")
    fooChild.bar('HelloWorld')