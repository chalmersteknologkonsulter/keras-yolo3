from matplotlib import pyplot as plt

model_versions = ['C:/Users/CTK-VR1/PycharmProjects/mAP/results/1', 'C:/Users/CTK-VR1/PycharmProjects/mAP/results/2', 'C:/Users/CTK-VR1/PycharmProjects/mAP/results/3']
xaxis = 'Number of frozen layers in model' #hat does the x-axis signify in the graph, i.e. layers frozen

mAP = [67,87,98]
ap_bird = [23,45,56]
ap_airplane = [98,87,76]
ap_drone = [54,65,76]

plt.plot(mAP, 'ro', label='mAP')
plt.plot(ap_airplane, 'b*', label='AP Airplane')
plt.plot(ap_drone, 'g*', label='AP Drone')
plt.plot(ap_bird, 'm*', label='AP Bird')
plt.xticks(rotation=90)
plt.xlabel(xaxis)
plt.ylabel('mAP (%)')
plt.legend()
plt.savefig('map_graph')
plt.show()
