# Written in micropython
import sensor, image, time, tf, math

width = 240
height = 240

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((width, height))
sensor.skip_frames(time=5000)

n = 2
m = 3
min_confidence = 0.4

def cal(step, size, ratio):
    return math.ceil(step * size / ratio)

labels, net = tf.load_builtin_model("fomo_face_detection")

colors = [
    (255,   0,   0),
    (  0, 255,   0),
    (255, 255,   0),
    (  0,   0, 255),
    (255,   0, 255),
    (  0, 255, 255),
    (255, 255, 255),
]

clock = time.clock()
clock.tick()

coordinate = []

while True:
    img = sensor.snapshot()

    for _w in range(n):
        for _h in range(m):
            croped = img.copy(roi=(cal(_w, width, n), cal(_h, height, m), cal(_w + 1, width, m), cal(_h + 1, height, m)))
            for i, detection_list in enumerate(net.detect(croped, thresholds=[(math.ceil(min_confidence * 255), 255)])):
                if (i == 0): continue
                if (len(detection_list) == 0): continue

                print("********** %s **********" % labels[i])
                for d in detection_list:
                    print(f"detected: {d}")

                    [x, y, w, h] = d.rect()

                    center_x = math.floor(x + (w / 2) + cal(_w, width, n))
                    center_y = math.floor(y + (h / 2) + cal(_h, height, m))
                    print(f"x {center_x}\ty {center_y}")
                    img.draw_circle((center_x, center_y, 12), color=colors[i], thickness=2)

