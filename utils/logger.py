import logging
import sys
import time


class Logger(object):
    def __init__(self, filename):
        self.start_time = time.time()
        self.logger = logging.getLogger("abcRec")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # write into file
        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        # show on screen
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(formatter)
        # add two Handler
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)

    def __del__(self):
        total_time = self.start_time - time.time()
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        self.info("Proceeding has finished, the total time:\t%02dh:%02dm:%02ds\n\n" % (h, m, s))

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


if __name__ == '__main__':
    log = Logger('abcRec_test.log')
    log.debug('debug')
    log.info('info')
    log.warning('警告')
    log.error('报错')
    log.critical('严重')
