from ultralytics import YOLO
import cv2

model = YOLO(r"C:\python_projects\helmet survilence\no_helmet.pt")

frame = cv2.imread(r"C:\python_projects\helmet survilence\dc-Cover-mdmk6mpl3ea84k1cn6pln9rrv5-20160306024001.Medi.jpeg")
results = model(frame)

names = model.names

motorcyclists = []
no_helmets = []
plates = []

# -------------------------------
# STEP 1: Separate detections
# -------------------------------
for r in results:
    boxes = r.boxes.xyxy
    classes = r.boxes.cls
    confs = r.boxes.conf

    for box, cls, conf in zip(boxes, classes, confs):
        label = names[int(cls)]
        x1, y1, x2, y2 = map(int, box)

        if conf < 0.5:
            continue

        if label == "motorcyclist":
            motorcyclists.append((x1, y1, x2, y2))

        elif label == "no-helmet":
            no_helmets.append((x1, y1, x2, y2))

        elif label == "plate":
            plates.append((x1, y1, x2, y2))

# -------------------------------
# STEP 2: Check relationships
# -------------------------------
def inside(inner, outer):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    return x1 > X1 and y1 > Y1 and x2 < X2 and y2 < Y2

# -------------------------------
# STEP 3: Apply logic
# -------------------------------
for rider in motorcyclists:

    rider_has_no_helmet = False
    rider_plate = None

    # check if no-helmet inside rider
    for nh in no_helmets:
        if inside(nh, rider):
            rider_has_no_helmet = True
            break

    if not rider_has_no_helmet:
        continue

    # find plate inside same rider
    for p in plates:
        if inside(p, rider):
            rider_plate = p
            break

    if rider_plate is None:
        continue

    # -------------------------------
    # FINAL RESULT (VIOLATION)
    # -------------------------------
    x1, y1, x2, y2 = rider
    px1, py1, px2, py2 = rider_plate

    # draw rider
    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.putText(frame, "NO HELMET RIDER", (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # draw plate
    cv2.rectangle(frame, (px1,py1), (px2,py2), (0,255,0), 2)

cv2.imwrite("output_detect.jpg", frame)
print("Done")