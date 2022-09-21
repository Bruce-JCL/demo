import get_android
#拉流
get_android.androidOb("ip_camera_config.json",1,960,540,4)
#原生
# get_android.androidOb("native_config.json",4,1920,1080,1)
#usb
# get_android.androidOb("uvc_config.json",6,1920,1080,1)
#拉流推流
# get_android.androidOb("ip_camera_encode_config.json",2,960,540,4)
#拉流推流存储
# get_android.androidOb("ip_camera_save_config.json",3,960,540,4)
#原生推流
# get_android.androidOb("native_encode_config.json",5,1920,1080,1)

#算法1
get_android.run(get_android.ai_worker1)
#算法2
# get_android.run(get_android.ai_worker2)


