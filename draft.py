from models import RoadSurfaceNet
from models import SideNet
from roadnet import RoadNet

net = RoadSurfaceNet()
model = net.get_model()

sidenet = SideNet()
side_model = sidenet.get_model()

road_net = RoadNet().get_model()

print(model.summary())
print(side_model.summary())
print(road_net.summary())

# print(road_net.get_layer('surface').get_layer('surface_output_1').get_weights())
