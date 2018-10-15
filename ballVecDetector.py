import cv2
import numpy as np
import math
import time


def loop():
    velocity_path = "./velocity_output"
    np.set_printoptions(threshold=np.inf)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 30, (640, 480))

    DURATION = 1.0
    LINE_LENGTH_ALL = 120
    LINE_LENGTH_GRID = 5
    GRID_WIDTH = 40
    CIRCLE_RADIUS = 2

    cap = cv2.VideoCapture(0)

    cv2.namedWindow('motion')
    ret, frame_next = cap.read()
    height, width, channels = frame_next.shape
    motion_history = np.zeros((height, width), np.float32)
    frame_pre = frame_next.copy()
    frame_next = frame_next[:,::-1]
    frame_number = 0
    point_x = 0
    point_y = 0

    detecting = False
    starttime = 0
    threshold = 1400000
    distance = 50
    detected_count = 0
    detected_rad_total = 0
    detected_rad_average = 0

    while True:
        frame_number += 1
        # フレーム間の差分計算
        color_diff = cv2.absdiff(frame_next, frame_pre)
        # グレースケール変換
        gray_diff = cv2.cvtColor(color_diff, cv2.COLOR_BGR2GRAY)
        # ２値化
        retval, black_diff = cv2.threshold(gray_diff, 30, 1, cv2.THRESH_BINARY)
        # プロセッサ処理時間(sec)を取得
        proc_time = time.clock()
        # モーション履歴画像の更新
        cv2.motempl.updateMotionHistory(black_diff, motion_history, proc_time, DURATION)
        # 古いモーションの表示を経過時間に応じて薄くする
        hist_color = np.array(np.clip((motion_history - (proc_time - DURATION)) / DURATION, 0, 1) * 255, np.uint8)
        # グレースケール変換
        hist_gray = cv2.cvtColor(hist_color, cv2.COLOR_GRAY2BGR)

        # モーションの変化率を計算
        movements_level = gray_diff.sum()
        # print(movements_level)

        # モーション履歴画像の変化方向の計算
        #   ※ orientationには各座標に対して変化方向の値（deg）が格納されます
        mask, orientation = cv2.motempl.calcMotionGradient(motion_history, 0.25, 0.05, apertureSize = 5)

        # 全体的なモーション方向を計算
        angle_deg = cv2.motempl.calcGlobalOrientation(orientation, mask, motion_history, proc_time, DURATION)
        # 全体の動きを黄色い線で描画
        cv2.circle(hist_gray,
                   (int(width / 2), int(height / 2)),
                   CIRCLE_RADIUS,
                   (0, 215, 0),
                   2,
                   16,
                   0)
        angle_rad = math.radians(angle_deg)

        cv2.line(hist_gray,
                 (int(width / 2), int(height / 2)),
                 (int(width / 2 + math.cos(angle_rad) * LINE_LENGTH_ALL), int(height / 2 + math.sin(angle_rad) * LINE_LENGTH_ALL)),
                 (0, 215, 0),
                 2,
                 16,
                 0)

        if movements_level >= threshold and not detecting:
            detected_count = 0
            detected_rad_total = 0
            detecting = True
            starttime = time.time()
            print("Detection!")

        if detecting:
            detected_count += 1
            detected_rad_total += angle_rad

        if movements_level < threshold and detecting:
            detecting = False
            detected_rad_average = detected_rad_total / detected_count
            endtime = time.time()-starttime
            detected_speed = (distance/100)/endtime
            print("Detection end.", detected_rad_average, "rad.",detected_speed, "m/s")
            with open(velocity_path, 'w') as f:
                f.write(str(detected_speed)+'\n')
                f.write(str(detected_rad_average)+'\n')

        # 各座標の動きを緑色の線で描画
        width_i = GRID_WIDTH
        while width_i < width:
            height_i = GRID_WIDTH
            while height_i < height:
                # cv2.circle(hist_gray,
                #             (width_i, height_i),
                #             CIRCLE_RADIUS,
                #             (0, 255, 0),
                #             2,
                #             16,
                #             0)
                # angle_deg = orientation[height_i - 1][width_i - 1]
                # if angle_deg > 0:
                #     angle_rad = math.radians(angle_deg)
                #     cv2.line(hist_gray,
                #             (width_i, height_i),
                #             (int(width_i + math.cos(angle_rad) * LINE_LENGTH_GRID), int(height_i + math.sin(angle_rad) * LINE_LENGTH_GRID)),
                #             (0, 255, 0),
                #             2,
                #             16,
                #             0)
                if frame_number % 2 == 0:
                    move_points = np.where(orientation > 10)
                    point_x = np.nanmean(move_points[0])
                    point_y = np.nanmean(move_points[1])
                    if str(point_x) == 'nan':
                        point_x = width // 2
                    if str(point_y) == 'nan':
                        point_y = height // 2

                    point_x = int(point_x)
                    point_y = int(point_y)

                cv2.circle(hist_gray,
                            (point_y, point_x),
                            CIRCLE_RADIUS,
                            (0, 0, 255),
                            2,
                            16,
                            0)
                try:
                    cv2.line(hist_gray,
                             (point_y, point_x),
                             (0, int(math.tan(angle_rad)*(-point_y)+point_x)),
                             (0, 215, 0),
                             2,
                             16,
                             0)
                except:
                    pass

                height_i += GRID_WIDTH
            width_i += GRID_WIDTH

        cv2.putText(hist_gray, str(angle_deg), (32, 32), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(hist_gray, str(-math.tan(angle_rad)), (32, 64), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        # モーション画像を表示
        cv2.imshow("motion", hist_gray)
        # 動画を保存
        out.write(hist_gray)
        # Escキー押下で終了
        if cv2.waitKey(1) == 0x1b:
            break
        # 次のフレームの読み込み
        frame_pre = frame_next.copy()
        ret, frame_next = cap.read()
        frame_next = frame_next[:,::-1]


    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
    return

loop()
