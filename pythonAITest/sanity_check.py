import cv2
# import matplotlib.pyplot as plt
#
# plt.imshow(x)
# plt.show()

x='C:\\Users\\jpg\Desktop\\pythonAITest\\new_data\\train\image\\ache-adult-depression-expression-41253_0.png'
y='C:\\Users\\jpg\Desktop\\pythonAITest\\new_data\\train\mask\\ache-adult-depression-expression-41253_0.png'

x = cv2.imread(x, cv2.IMREAD_COLOR)
y = cv2.imread(y, cv2.IMREAD_COLOR)
y=y*255
cv2.imshow("x",x)
cv2.imshow("y",y)
key = cv2.waitKey(0)

# Check if the 'Esc' key (27) or 'q' key (113) was pressed
if key == 27 or key == 113:
    cv2.destroyAllWindows()


import tensorflow as tf

# Check the list of available GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))

# You can also limit GPU memory growth (optional)
for gpu in physical_devices:
    tf.config.experimental.set_memory_growth(gpu, True)