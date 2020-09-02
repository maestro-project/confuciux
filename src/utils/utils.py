def get_platform_ratio(platform):
    if platform == "cloud":
        return 0.5
    elif platform == "IoT":
        return 0.1
    elif platform == "eIoT":
        return 0.05
    else:
        print("Please choose from [cloud, IoT, eIoT].")