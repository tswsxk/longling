#!/usr/bin/env python
# encoding: utf-8

import functools
import logging
import os
import sys
import time

from framework.keyScene.common import get_conf
from framework.keyScene.common.convert import hex2binary
from framework.keyScene.common.get_scenecut_frame import get_scenecut_frame
from framework.keyScene.common.image_mosaic import get_offline_mosaic_client
from pyutil.program.fmtutil import fmt_exception

from longling.framework.keyScene.common.file_tools import ensure_dir_exists, remove_dir

conf = get_conf()

min_number = 30
tmp_store_path = conf.get('scene_cut_tmp_path', './output/insight_store/scenecut_store')

#定义一个时间统计的修饰函数，用来统计函数运行时间
def time_report(info = "None"):
    def _time_report(fn):
        @functools.wraps(fn)
        def warppendFunc(*args, **kwargs):
            start_time = time.time()
            logging.info("vid = %s: SCENE_CUT_SESSION-%s started!"%(args[0].vid, info))
            res = fn(*args, **kwargs)
            end_time = time.time()
            logging.info("vid = %s: SCENE_CUT_SESSION-%s finished! consume time: %fs"
                         %(args[0].vid, info, end_time-start_time))
            return res
        return warppendFunc
    return _time_report

class SceneCut(object):

    @classmethod
    def __init__(self,  vid=None, video_path=None, provider=None,
                 scene_cut_tmp_path=tmp_store_path):
        """
        Init the base info
        :vid: vid of video
        :video: fullpath of local video
        :provider: the provider name
        :base_dir: the tmp_dir path of scenecut
        """
        self.vid = vid.lower()
        self.video = video_path
        self.basedir = os.path.join(scene_cut_tmp_path, vid)
        self.provider = provider
        ensure_dir_exists(self.basedir)

    @classmethod
    @time_report("get scenecut frames")
    def get_capture_frames(self):
        """
        get scene_cut frames
        """
        thumbnail_dir = self.basedir
        try:
            # hotsoon user only capture 5 frames
            if self.provider in ["hotsoon", "hotsoon_logo", 'hypstar', 'hypstar_logo']:
                print "vid=%s: hotsoon user[%s] only capture 5 frames!" %(self.vid, self.video_info.provider)
                param = {'input': self.video, "tmp_dir": thumbnail_dir, "max_frames": 5}
            else:
                param = {'input': self.video, "tmp_dir": thumbnail_dir}
        except:
            param = {'input': self.video, "tmp_dir": thumbnail_dir}
        # 获取需要截的帧
        pic_info = get_scenecut_frame(param)
        if pic_info and 'capture_frames' in pic_info:
            frames = pic_info['capture_frames']
            logging.info("vid= %s, scene cut frames: %s" % (self.vid, str(frames)))
            self.ill_flag = pic_info.get('ill', None)
                #self.save_video_ill(pic_info['ill'])
        else:
            frames = []
            raise Exception("vid= %s, err_msg: %s" % (self.vid, pic_info['err_msg']))

        print 'vid=%s: final scene cut frames: %s' % (self.vid, str(frames))
        return frames

    @classmethod
    @time_report("upload frames to mosaic")
    def upload_frames(self, frames):
        """
        upload frames to mosaic
        :@params frames: a list if frame_no
        """
        thumbnail_dir = self.basedir
        image_dict = dict()
        for frame in frames:
            thumbnail = '%s/%s.jpg' %(thumbnail_dir, frame)
            if os.path.exists(thumbnail):
                try:
                    image_data = None
                    with open(thumbnail, 'r') as image_file:
                        image_data = image_file.read()
                    if not image_data:
                        logging.error('vid=%s: read thumbnail fail' % self.vid)
                        continue
                    res = get_offline_mosaic_client().post_raw(image_data, 'image/jpeg')
                    uri = res.get('key', '')
                    if uri:
                        image_dict.update({frame: uri})
                        logging.info('vid=%s: save thumbnail %s to mosaic success' % (self.vid, frame))
                    else:
                        logging.warning('vid=%s: save thumbnail %s to mosaic fail' % (self.vid, frame))
                except Exception as e:
                    logging.error('vid=%s: save thumbnail %s to mosaic fail: %s' % (
                        self.vid, frame, fmt_exception(e)))
        return image_dict

    @classmethod
    @time_report('save relation of scenecut_frames and video to mysql')
    def save_map_relation(self, image_dict):
        """
        save the video and scencut_frames relationship to db
        """
        for k in image_dict.keys():
            frame_no = int(k)
            img_uri = image_dict.get(k)
            kwargs = {'img_uri': img_uri}
            try:
                VideoSceneCutDal.update_item(ref_id=self.vid, frame_no=frame_no, **kwargs)
                logging.info('vid=%s: save relation of thumbnail %d to mysql succeed'
                                % (self.vid, frame_no))
            except Exception as e:
                err_msg =  "vid=%s: save relation of thumbnail %d to mysql Faild: %s" % (self.vid, frame_no, fmt_exception(e))
                logging.warning(err_msg)
                raise Exception(err_msg)

    @classmethod
    @time_report('save video_ill info to mysql')
    def save_video_ill(self, ill_flag):
        """
        save ill_flag of video to db
        """
        if self.vid:
            # update(if exists) or create(if not exists) a new item
            ret = VideoArtifactDetectDal.update_item(vid=self.vid, ill_flag=ill_flag)
            if ret:
                logging.info('vid=%s: save ill_flag %d to mysql succeed' % (self.vid, ill_flag))
            else:
                logging.warning('vid=%s: save ill_flag %d to mysql failed' % (self.vid, ill_flag))
            return True

        logging.warning('no vid=%s' % (self.vid))
        return False

    @classmethod
    def save_to_db(self, image_dict):
        self.save_map_relation(image_dict)
        if self.ill_flag:
            self.save_video_ill(self.ill_flag)

    @classmethod
    def scene_cut(self, force = False):
        """
        scene_cut session
        :params force: when force is True, the session will begin regardless it have been done or not
        :return: res_flag(False for faild, True for succeed), res_msg;
        """
        if not self.vid:
            raise Exception("SCENE_CUT_SESSION-vid is None, Please provide vid")
        if hex2binary(self.vid) == 0:
            raise Exception("SCENE_CUT_SESSION-vid is illegal, vid must be 32bit hex")
        if not self.video:
            raise Exception("SCENE_CUT_SESSION-video_path is None, Please privide video_path")
        if (not force) and self.check_exists():
            logging.info("vid=%s: SCENE_CUT_SESSION-the video have already been scenecuted before!" % self.vid)
            return None

        #获取场景变换截图
        try:
            frames = self.get_capture_frames()
        except Exception as e:
            raise Exception(e)

        #截图上传
        image_dict = self.upload_frames(frames=frames)

        #存储关系上传
        self.save_to_db(image_dict)

        #rm tmp_path
        remove_dir(self.basedir)

        if len(image_dict) < len(frames)*0.9: #存储成功率低于90%认为影响结果，抛出异常
            raise Exception("vid=%s, SCENE_CUT_SESSION-save frames to mosaic unsuccessful!" % self.vid)

    @classmethod
    def check_exists(self):
        if VideoSceneCutDal.get_items_by_ref_id(ref_id = self.vid):
            return True
        else:
            return False

if __name__ == '__main__':
    """
    test:
    :@input format: python scene_cut.py vid video_path provider(option)
    """
    real_run = True
    leng = len(sys.argv)
    if leng == 3:
        vid = sys.argv[1]
        video_path = sys.argv[2]
        print vid, video_path
        if real_run:
            SceneCut(vid=vid, video_path=video_path).scene_cut(force=True)
    elif leng == 4:
        vid = sys.argv[1]
        video_path = sys.argv[2]
        provider = VideoUser.sys.argv[3]
        print vid, video_path, provider
        if real_run:
            SceneCut(vid=vid, videp_path=video_path, provider = provider).scene_cut(force=True)
    else:
        print "illegal input format, please run as: python scene_cut.py vid video_path provider(option)"

