import cv2
import pafy
import setup
import logic

FV_MODEL = setup.get_fv_model()
NBR_MODEL = setup.get_nbr_model()


def process_image(img):
    if img.shape == (224, 224, 3):
        return logic.get_label([img], nbr_model=NBR_MODEL, fv_model=FV_MODEL)
    else:
        return "nothing"
    # print(img)
    # return "Elephant"


def main(source: str = "webcam"):
    if source == "webcam":
        cap = cv2.VideoCapture(1)
    else:
        video = pafy.new(source)
        best = video.getbest(preftype="mp4")
        cap = cv2.VideoCapture(best.url)
    cap.set(cv2.CAP_PROP_FPS, 36)

    while True:
        ret, img = cap.read()
        if ret:
            img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_AREA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            text = process_image(img)
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
        cv2.imshow("WEBCAMERA", img)
        if cv2.waitKey(10) == 27:  # Клавиша Esc
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main("https://youtu.be/Sm4oH-CSaEM")
    # main()
