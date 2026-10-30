from damage_ai.inference import predict
class T:
    def __init__(self,x): self.x=x
    def item(self): return self.x
    def tolist(self): return self.x
class Box:
    cls=T(0); conf=T(.9); xyxy=[T([1,2,3,4])]
class Result:
    names={0:"scratch"}; boxes=[Box()]
class Model:
    def predict(self,**kwargs): return [Result()]
def test_prediction_format():
    r=predict(Model(),"fake.jpg")
    assert r[0]["label"]=="scratch"
    assert r[0]["bbox_xyxy"]==[1,2,3,4]
