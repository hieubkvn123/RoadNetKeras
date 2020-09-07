from models import RoadSurfaceNet

net = RoadSurfaceNet()
model = net.get_model()

print(model.summary())
