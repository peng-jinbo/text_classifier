from search import *
from tqdm import tqdm
from multiprocessing.pool import Pool


def scool_normalize(ori_l):
    nor_l = []
    with Pool(MP_NUM) as pool:
        for i in tqdm(pool.imap(predict, [calibrated(j) for j in ori_l]), total=len(ori_l)):
            nor_l.append(i)
    return nor_l


if __name__ == "__main__":
    ori_list = ["Shanghaijiaotonguniversity", "Shanghai jiao tong university", "Shanghai Electricity College", "new york university"] * 10
    print(scool_normalize(ori_list))