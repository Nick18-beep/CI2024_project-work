import numpy as np


def f1(x: np.ndarray) -> np.ndarray: 
    return np.sin(x[0])


def f2(x: np.ndarray) -> np.ndarray: 
    return (np.square(93.65140858898903 - np.square(x[0] + x[2])) *
            (((x[1] + x[0]) * (60.47785699965641 - -53.47835550884468)) - 
             ((x[2] + x[0]) * (-71.74007332603995 + -96.15173552509577)))) + \
           (x[0] * ((np.square(np.square(22.052833372136263))) * 
                     ((x[1] * x[0]) / -11.39694535089464)) + 
            np.square(x[2] - np.sin(30.54037427125394)))


def f3(x: np.ndarray) -> np.ndarray: 
    return (((((x[0] * x[0]) - (x[1] * (x[1] * x[1]))) + (x[0] * x[0])) - np.sin(-8.96354622603981)) - (x[2] * 3.4999999952851786)) - -3.5549484801938522


def f4(x: np.ndarray) -> np.ndarray: 
    return ((3.2794167237785796 + (x[0] * np.cos(4.621354253617349))) + 
            (6.9999998655622075 * np.cos(x[1])))


def f5(x: np.ndarray) -> np.ndarray: 
    return ((np.exp((x[1] / x[1])) + (np.exp(0.37750334939704544) + (x[1] / (-1.8246868030183787 / x[0])))) / 
            np.exp((np.exp(3.1900906070573303) + ((x[1] / 2.9208916743572377) / (-1.8246868030183787 / x[0])))))


def f6(x: np.ndarray) -> np.ndarray: 
    return (((x[0] * -0.6945204040717841) - (x[1] * -0.6945204040959478)) + x[1])


def f7(x: np.ndarray) -> np.ndarray: 
    return (np.square(np.log(np.square((((np.sin(np.sin(x[0])) * (0.19222894326264073 * x[1])) * x[0]) * 
                                         ((x[1] + -0.002222450188612113) - x[0]))))) * 
            (x[0] * ((((x[1] + np.square(((x[0] * -0.9340113825923886) * (0.9340113825923886) * 
                                   (0.14251173977126474 - 0.16975453430987697)))) * 0.3947100768049452) * x[0]) * 
             (0.016197739459344884 + x[1]))))


def f8(x: np.ndarray) -> np.ndarray: 
    return (((np.square(np.square(x[5]))) + ((x[1] - x[5]) - ((np.square(x[4]) + np.square(x[4])) * (x[5] + x[5])))) + 
            np.square((np.square(x[5]) + np.square(x[5])))) * x[5]