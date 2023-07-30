import win32con
import win32gui
import camera.OAK_D_api as oak
import cv2
import torch
import torchvision
import src.models.YOLO as YOLO
import src.utils.utils as utils


def run_camera(fps=60, width=448, height=448, model=None):
    oak_d = oak.OAK_D(fps=fps, width=width, height=height)
    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        img = frame.copy()
        frame = torchvision.transforms.functional.to_tensor(frame)
        frame = torch.unsqueeze(frame, 0)
        prediction = model(frame)
        prediction = prediction.reshape((7, 7, 5 * 2 + 20))
        img = utils.draw_image_predictions(img, predictions=prediction)

        cv2.imshow("YOLOnaut", img)
        hwnd = win32gui.FindWindow(None, "YOLOnaut")
        icon_path = "camera/YOLOnaut.ico"
        win32gui.SendMessage(hwnd, win32con.WM_SETICON, win32con.ICON_BIG,
                             win32gui.LoadImage(None, icon_path, win32con.IMAGE_ICON, 0, 0,
                                                win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE))

        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    model = YOLO.YOLO()
    # TODO: load model on GPU
    model.load_state_dict(
        torch.load('models/initial_model.pth', map_location=torch.device('cpu')))

    run_camera(model=model)
