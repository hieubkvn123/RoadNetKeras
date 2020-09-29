import pandas as pd
import matplotlib.pyplot as plt

TRAINING_LOG = 'training.log.csv'
data_frame = pd.read_csv(TRAINING_LOG).dropna()

columns = [
    'epoch',
    'loss',
    'val_loss',
    'surface_final_output_loss',
    'val_surface_final_output_loss',
    'edge_final_output_loss',
    'val_edge_final_output_loss',
    'line_final_output_loss',
    'val_line_final_output_loss',
    'surface_final_output_my_mean_iou',
    'val_surface_final_output_my_mean_iou',
    'edge_final_output_my_mean_iou',
    'val_edge_final_output_my_mean_iou',
    'line_final_output_my_mean_iou',
    'val_line_final_output_my_mean_iou'
]

data_frame = data_frame[columns]
data_frame['epoch'] = list(range(len(data_frame['epoch'])))

fig1, ax1 = plt.subplots(2,2, figsize=(15,15))
fig2, ax2 = plt.subplots(3,1, figsize=(10, 20))
plt.style.use('dark_background')
plt.rcParams['savefig.facecolor'] = 'white'

ax1[0][0].plot(data_frame['epoch'], data_frame['loss'], color='blue', label='Total Loss', linewidth=2)
ax1[0][0].plot(data_frame['epoch'], data_frame['val_loss'], color='red', label='Total Validation Loss', linewidth=2)
ax1[0][0].set_title('Total Losses')
ax1[0][0].legend()

ax1[0][1].plot(data_frame['epoch'], data_frame['surface_final_output_loss'], color='blue', label='Surface Loss', linewidth=2)
ax1[0][1].plot(data_frame['epoch'], data_frame['val_surface_final_output_loss'], color='red', label='Surface Validation Loss', linewidth=2)
ax1[0][1].set_title('Total losses (Surface prediction)')
ax1[0][1].legend()

ax1[1][0].plot(data_frame['epoch'], data_frame['edge_final_output_loss'], color='blue', label='Edge Loss', linewidth=2)
ax1[1][0].plot(data_frame['epoch'], data_frame['val_edge_final_output_loss'], color='red', label='Edge Validation Loss', linewidth=2)
ax1[1][0].set_title('Total losses (Edge prediction)')
ax1[1][0].legend()

ax1[1][1].plot(data_frame['epoch'], data_frame['line_final_output_loss'], color='blue', label='Line Loss', linewidth=2)
ax1[1][1].plot(data_frame['epoch'], data_frame['val_line_final_output_loss'], color='red', label='Line Validation Loss', linewidth=2)
ax1[1][1].set_title('Total losses (Center-line prediction)')
ax1[1][1].legend()

ax2[0].plot(data_frame['epoch'], data_frame['surface_final_output_my_mean_iou'], color='blue', label='Surface MeanIOU', linewidth=2)
ax2[0].plot(data_frame['epoch'], data_frame['val_surface_final_output_my_mean_iou'], color='red', label='Surface Validation MeanIOU', linewidth=2)
ax2[0].set_title('MeanIOU (Surface prediction)')
ax2[0].legend()

ax2[1].plot(data_frame['epoch'], data_frame['edge_final_output_my_mean_iou'], color='blue', label='Edge MeanIOU', linewidth=2)
ax2[1].plot(data_frame['epoch'], data_frame['val_edge_final_output_my_mean_iou'], color='red', label='Edge Validation MeanIOU', linewidth=2)
ax2[1].set_title('MeanIOU (Edge prediction)')
ax2[1].legend()

ax2[2].plot(data_frame['epoch'], data_frame['line_final_output_my_mean_iou'], color='blue', label='Center-line MeanIOU', linewidth=2)
ax2[2].plot(data_frame['epoch'], data_frame['val_line_final_output_my_mean_iou'], color='red', label='Center-line Validation MeanIOU', linewidth=2)
ax2[2].set_title('MeanIOU (Center-line prediction)')
ax2[2].legend()

for ax_ in ax1:
    for ax in ax_:
        ax.set_facecolor('black')
        ax.grid(color='gray')

for ax in ax2:
    ax.set_facecolor('black')
    ax.grid(color='gray')

fig1.savefig('training_results/losses.png')
fig2.savefig('training_results/mean_iou.png')

plt.show()
