#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on 2016-7-20

@author: hejianqiang
"""

import os
import commands
# import subprocess
# import time

import re
import json

import copy
import locale

from os.path import join, isfile
import sys

prefix_dir = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(prefix_dir)

console_encoding = locale.getdefaultlocale()[1] or 'UTF-8'


TR_VERSION = '2017-3-10'
FFMPEG_VERSION = 3.1

# format = re.compile(r'\[([0-9]{2}\/[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}).*(situation [ABCD]).*')

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg2.6.1')):
    FFMPEG2 = join(prefix_dir, '../../vendor/bin/ffmpeg2.6.1')
else:
    FFMPEG2 = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg2.6.1'

if isfile(join(prefix_dir, '../../vendor/bin/ffprobe2.6.1')):
    FFPROBE2 = join(prefix_dir, '../../vendor/bin/ffprobe2.6.1')
else:
    FFPROBE2 = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffprobe2.6.1'

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg2.8.5.webp')):
    FFMPEG_WEBP = join(prefix_dir, '../../vendor/bin/ffmpeg2.8.5.webp')
else:
    FFMPEG_WEBP = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg2.8.5.webp'

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg3.1.2')):
    FFMPEG = join(prefix_dir, '../../vendor/bin/ffmpeg3.1.2')
else:
    FFMPEG = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg3.1.2'

if isfile(join(prefix_dir, '../../vendor/bin/ffprobe3.1.2')):
    FFPROBE = join(prefix_dir, '../../vendor/bin/ffprobe3.1.2')
else:
    FFPROBE = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffprobe3.1.2'

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg_nofree')):
    FFMPEG_NO = join(prefix_dir, '../../vendor/bin/ffmpeg_nofree')
else:
    FFMPEG_NO = FFMPEG  # '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg_nofree'

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg3.3')):
    FFMPEG33 = join(prefix_dir, '../../vendor/bin/ffmpeg3.3')
else:
    FFMPEG33 = FFMPEG

if isfile(join(prefix_dir, '../../vendor/bin/ffprobe3.3')):
    FFPROBE33 = join(prefix_dir, '../../vendor/bin/ffprobe3.3')
else:
    FFPROBE33 = FFPROBE

if isfile(join(prefix_dir, '../../vendor/bin/ffmpeg3.0_fix')):
    FFMPEG_FIX = join(prefix_dir, '../../vendor/bin/ffmpeg3.0_fix')
else:
    FFMPEG_FIX = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg3.0_fix'

if isfile(join(prefix_dir, '../../vendor/bin/ffprobe3.0_fix')):
    FFPROBE_FIX = join(prefix_dir, '../../vendor/bin/ffprobe3.0_fix')
else:
    FFPROBE_FIX = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffprobe3.0_fix'

if isfile(join(prefix_dir, '../../vendor/bin/MP4Box_1.3')):
    MP4Box = join(prefix_dir, '../../vendor/bin/MP4Box_1.3')
else:
    MP4Box = '/opt/tiger/toutiao/app/videoarch/vendor/bin/MP4Box_1.3'

if isfile(join(prefix_dir, "../../vendor/bin/ffmpeg3.1.2_pgc")):
    FFMPEG_PGC = join(prefix_dir, "../../vendor/bin/ffmpeg3.1.2_pgc")
else:
    FFMPEG_PGC = '/opt/tiger/toutiao/app/videoarch/vendor/bin/ffmpeg3.1.2_pgc'

if isfile(join(prefix_dir, '../../vendor/bin/webpmux')):
    WEBP_MUX = join(prefix_dir, '../../vendor/bin/webpmux')
else:
    WEBP_MUX = '/opt/tiger/toutiao/app/videoarch/vendor/bin/webpmux'

CRF_DEFAULT = 28  # FIXME ffmpeg3.1.2默认25便可以, 但是如果有对比度增强滤波,会导致码率增加,则设为26

MAX_VOLUME_THRES = -7.5
MAX_VOLUME_ADD_THRES = 1.5 - MAX_VOLUME_THRES

MEAN_VOLUME_DIFF_DTHRES = 8.0  # 9.0 2ch-1ch>thres
MAX_VOLUME_1CH_THRES = -17.5  # -10.5

No_volume_value = -999.9
mean_volume_pat = re.compile(r'mean_volume:\s+([\d.-]+)\s+dB')
max_volume_pat = re.compile(r'max_volume:\s+([\d.-]+)\s+dB')

size_pat = re.compile(r'[0-9]{2,4}x[0-9]{2,4}')  # 640x360 # {2~4} num of digital x {2~4} num of digital

support_list = ["AVI", "WMA", "RMVB", "FLASH", "MP4", "GP3", "GIF", "MPEG", "FLV", "ASF", "MPEGTS", "RM", "WEBP", "AAC", "MP3", "FLA", "WAV", "M4A", "AMR"]

sufset = set(['.txt', '.sh', '.py', '.log', '.ass', '.ifo', '.bup', '.bat', '.pyc', '.srt', '.ds_store'])
videoset = set(['.mkv','.mp4','.avi','.flv','.f4v','.ts','.wmv','.mpg','.vob','.m2ts','.3gp', '.mov','.mpeg','.m4v','.mxf', ''])

picset = set(['.jpg', '.png', '.jpeg', '.jpeg2000', 'mjpeg', '.bmp', '.gif', '.tiff', '.webp', ''])

'''
means of err_code:
0: ok
1: fixed
2: mobile no sound
3: audio volume too small
4: only has video stream, no audio stream

-1: fixed failed or maybe dead loop, some other system problems. need re unpload
-2: input file size is < 5KB, or duration < 1 sec
-3: input format is not video, or unrecognised
-4: has no video stream, may only has audio;or only picture
-9: unknown
'''
ffmpeg_pic = FFMPEG33

# for jpg png
def ff_get_images(filename, tmp_dir='tmp', capture_num=7, fps=0.5, vf=None, width=-1, height=-1, suffix='.jpg', ss=0.0, limit=0, prefix='', short=0):
    if not os.path.isdir(tmp_dir):
        os.system("mkdir -p '%s'" % tmp_dir)

    filters = ''
    if width >= 16 or height >= 16:
        filters = ',scale=%d:%d' % (width, height)

    if vf is None:
        vf = {}
    if vf and 'vf' in vf:
        filters += ',%s' % vf['vf']

    if ss > 0.1:
        start_time = '-ss %s' % ss
    else:
        start_time = ''

    cap = ''
    if capture_num > 0:
        try:
            if capture_num > 30:
                # if more than 10 minutes, down to 1frame/10sec
                duration = get_video_duration(filename)
                if float(duration) > 600 and float(fps) > 0.1:
                    fps = 0.1
        except:
            pass

        cap = '-vframes %d' % int(capture_num)

    if suffix not in ['.jpg', 'jpeg', 'png']:
        print "must be jpg or png"
        suffix = '.jpg'

    if short:
        outfile = os.path.join(tmp_dir, prefix + r'%02d' + suffix)  # 01~99
    else:
        outfile = os.path.join(tmp_dir, prefix + r'%04d' + suffix)  # 0001~9999
    #  -pix_fmt yuvj420p
    cmd_str = "%s -v quiet %s -i '%s' -sn -an -f image2 -q:v 2 %s -vf 'fps=fps=%s%s' -y '%s' " \
              % (ffmpeg_pic, start_time, filename, cap, fps, filters, outfile)
    # print cmd_str
    ret, _ = commands.getstatusoutput(cmd_str)
    pic_list = get_dir_pics_list(tmp_dir, prefix='', suffix=suffix, limit=limit)
    return pic_list


# for dynamic webp
def ff_get_pictures(filename, tmp_dir='tmp', capture_num=11, fps=25, starttime=1, vf=None, width=-1, height=-1, suffix='.webp'):
    if not os.path.isdir(tmp_dir):
        os.system("mkdir -p '%s'" % tmp_dir)

    filters = ''
    if width >= 16 or height >= 16:
        filters = ',scale=%d:%d' % (width, height)

    if vf is None:
        vf = {}
    if vf and 'vf' in vf:
        filters += ',%s' % vf['vf']

    if capture_num < 1:
        capture_num = 1
    if fps < 1:
        fps = 1

    outfile = os.path.join(tmp_dir, r'%03d' + suffix)  # %%d 001~999
    ext_filter = ''
    if suffix.lower() == '.png' or suffix.lower() == '.jpg':
        ext_filter = '-f image2 -q:v 2'

    duration = capture_num/(fps+0.01)
    cmd_str = "%s -ss %s -v quiet -i '%s' -t %.2f -an -sn %s -vf 'fps=fps=%s%s' -y '%s' "\
              % (ffmpeg_pic, starttime, filename, duration, ext_filter, fps, filters, outfile)
    # print cmd_str
    ret, _ = commands.getstatusoutput(cmd_str)
    pic_list = get_dir_pics_list(tmp_dir, prefix='', suffix=suffix)
    return pic_list


def ff_get_first_pic(input_filename, tmp_dir, width=-1, height=-1, suffix='', ext='.jpg'):
    if not os.path.isdir(tmp_dir):
        os.system("mkdir -p '%s'" % tmp_dir)

    if suffix:
        first_jpg = os.path.join(tmp_dir, '%s.jpg' % suffix)
    else:
        first_jpg = os.path.join(tmp_dir, '0.jpg')

    filters = ''
    if width >= 16 or height >= 16:
        filters = '-vf scale=%d:%d' % (width, height)

    # -pix_fmt yuvj420p
    cmd_first = "%s -v quiet -ss 0 -i '%s' -an -sn -q:v 2 -vframes 1 %s -y '%s'" % (ffmpeg_pic, input_filename, filters, first_jpg)
    os.system(cmd_first)
    if os.path.isfile(first_jpg):
        return first_jpg
    return ''


def get_last_pic(input_filename, tmp_dir, duration, offset=0.2, width=-1, height=-1, suffix='', ext='.jpg'):
    basename0 = os.path.basename(input_filename)
    basename0 = basename0[:basename0.rfind('.')]
    basename0 = basename0.strip()

    last_pic_dir = os.path.join(tmp_dir, "last_%s" % basename0)
    os.system("mkdir -p '%s'" % last_pic_dir)
    jpg_lasts = os.path.join(last_pic_dir, '%04d.jpg')
    if suffix:
        last_jpg = os.path.join(tmp_dir, '%s.jpg' % suffix)
    else:
        last_jpg = os.path.join(tmp_dir, '9999.jpg')

    filters = ''
    if width >= 16 or height >= 16:
        filters = '-vf scale=%d:%d' % (width, height)

    try:
        duration = float(duration)
        ss = duration - offset
        ss = ss if ss > 0 else 0
        cmd_last_pics = "%s -hide_banner -ss %s -i '%s' -an -sn -q:v 2 -vframes 100 %s -y '%s'" % (ffmpeg_pic, ss, input_filename, filters, jpg_lasts)

        # print cmd_last_pics
        ret, _ = commands.getstatusoutput(cmd_last_pics)
        pic_list = get_dir_pics_list(last_pic_dir, prefix='', suffix='.jpg')
        if pic_list and len(pic_list) > 0:
            # print pic_list
            os.system("mv '%s' '%s'" % (pic_list[-1], last_jpg))
        elif ret or 0 <= offset < 2:  # and len(pic_list) == 0:
            offset += offset
            # print "the original duration offset cannot capture frames, double it, offset=%s, and contiune try" % offset
            last_jpg = get_last_pic(input_filename, tmp_dir, duration, offset=offset, width=width, height=height, suffix=suffix)

    except Exception as e:
        print "get_last_pic err=%s" % str(e)

    if os.path.isdir(last_pic_dir):
        os.system("rm -rf '%s'" % last_pic_dir)
    if os.path.isfile(last_jpg):
        return last_jpg
    return ''


def ff_get_last_pic(input_file, out_dir, width=-1, height=-1, suffix='', ext='.jpg', last_num=2):
    duration, input_fps = get_video_duration_fps(input_file)
    if input_fps < 0.1 and duration > 7200:
        print 'video: %s may has problem, duration=%s, fps=%s, only get first frames replace' % (input_file, duration, input_fps)
        return ff_get_first_pic(input_file, out_dir, width=width, height=height, suffix=suffix)

    if input_fps > 0:
        offset = 2.0 / input_fps  # 1.0*last_num
        offset = offset if offset > 0.02 else 0.02
    else:
        offset = 0.0

    # last_poc = int(duration*input_fps)
    # suffix = "%s%s" % (suffix, last_poc)
    return get_last_pic(input_file, out_dir, duration, offset=offset, width=width, height=height, suffix=suffix)


def av_interleave(inputfile, tmp_dir=''):
    if not os.path.isfile(inputfile) or not os.path.isfile(MP4Box) or os.path.getsize(inputfile) < 10000:  # <10k no need to do av interlace
        print "mp4box or file not exist"
        return

    tmp_file = inputfile + '_box.mp4'
    if os.path.isfile(tmp_file):
        tmp_file = inputfile + '_box_n.mp4'

    if tmp_dir and os.path.isdir(tmp_dir):
        cmd_str = "%s -add '%s' -tmp '%s' -strict-error -new '%s' " % (MP4Box, inputfile, tmp_dir, tmp_file)
    else:
        cmd_str = "%s -add '%s' -strict-error -new '%s' " % (MP4Box, inputfile, tmp_file)

    ret, _ = commands.getstatusoutput(cmd_str)  # ret = subprocess.call(cmd_str, shell=True, stdout=None, stderr=None)
    if ret == 0 and os.path.isfile(tmp_file):
        # os.system("rm -f '%s' " % inputfile)
        os.system("mv -f '%s' '%s' " % (tmp_file, inputfile))
    else:
        print 'av_interleave has problem, no change to video, MP4Box=%s, ret=%s!' % (MP4Box, ret)


def change_dyn_ffmpeg(inputfile):
    ret = 0
    if inputfile and os.path.isfile(inputfile):
        tmp_file = inputfile + '_box.mp4'
        cmd_str = "%s -add '%s' -strict-error -new '%s' " % (MP4Box, inputfile, tmp_file)
        ret, info = commands.getstatusoutput(cmd_str)
        if 'Unknown box type' in info or 'has error' in info or 'does not start with sync marker' in info or 'invalid in container' in info:
            ret = 1
        if tmp_file and os.path.isfile(tmp_file):
            os.system("rm -f '%s'" % tmp_file)

    return ret


def check_ffbug_avi(inputfile):
    ret = 0
    if inputfile and os.path.isfile(inputfile):
        tmp_file = inputfile + '_check.mp4'
        cmd_str = "%s -i '%s' -preset ultrafast -vframes 100 -y '%s' " % (FFMPEG33, inputfile, tmp_file)
        ret, info = commands.getstatusoutput(cmd_str)
        if 'Output file #0 does not contain any stream' in info:
            ret = 1
        if tmp_file and os.path.isfile(tmp_file):
            os.system("rm -f '%s'" % tmp_file)

    return ret


def ff_add_comments(inputfile, metadata=None):
    # metadata = {'vid': '123456', 'vr_type': 1}
    if not metadata or not os.path.isfile(inputfile):
        return

    new_file = inputfile + "_new_vid.mp4"
    vid = metadata.get('vid', '')
    vr_type = metadata.get('vr_type', '')

    comment = ''
    if vid:
        comment += 'vid:%s' % vid
        if vr_type:
            comment += ',vr_type:%s' % vr_type

        cmd_ff = "%s -i '%s' -c copy -movflags +faststart -metadata comment=%s -y '%s'" % (FFMPEG_NO, inputfile, comment, new_file)
        ret, _ = commands.getstatusoutput(cmd_ff)
        if ret == 0 and os.path.isfile(new_file):
            cmd_cp = "mv -f '%s' '%s' " % (new_file, inputfile)
            ret2, _ = commands.getstatusoutput(cmd_cp)


def create_logo_mask(param, width, height):
    if not param or 'pos' not in param or 'logo_mask_file' not in param:
        return None

    pos = param['pos']
    logo_mask_w = param['w']
    logo_mask_h = param['h']
    logo_mask_file = param['logo_mask_file']

    if logo_mask_w < 16 or logo_mask_h < 16:
        print 'logo wxh=%sx%s is err, guess as 640x360' % (logo_mask_w, logo_mask_h)
        logo_mask_w = 640
        logo_mask_h = 360

    # logo_num = len(pos)
    black_mask = join(prefix_dir, "black_640x360.png")
    white_mask = join(prefix_dir, "white_640x360.png")

    if not (os.path.isfile(black_mask) and os.path.isfile(white_mask)):
        return None

    # print pos
    for i in range(len(pos)-1, -1, -1):  # reverse
        if pos[i] and pos[i][2] <= 0 or pos[i][3] <= 0:
            del pos[i]
            # print 'pos paramters is incorrect, will be delete'

    if len(pos) == 0:
        return None

    for i in range(0, len(pos)):
        if pos[i][0] <= 2:
            pos[i][0] = 2
        if pos[i][1] <= 2:
            pos[i][1] = 2
        if pos[i][0] + pos[i][2] >= logo_mask_w - 1:
            pos[i][2] = logo_mask_w - 2 - pos[i][0]
            print 'pos[', i, ']w->', pos[i][2]
        if pos[i][1] + pos[i][3] >= logo_mask_h - 1:
            pos[i][3] = logo_mask_h - 2 - pos[i][1]
            print 'pos[', i, ']h->', pos[i][3]

    logo_num = len(pos)

    if logo_num == 1:
        vf = "[0:v]scale=w=%d:h=%d[a];[1:v]scale=w=%d:h=%d[o];[a][o]overlay=%d:%d" \
             % (logo_mask_w, logo_mask_h, pos[0][2], pos[0][3], pos[0][0], pos[0][1])
    elif logo_num == 2:
        vf = "[0:v]scale=w=%d:h=%d[a];[1:v]scale=w=%d:h=%d[o];[a][o]overlay=%d:%d[b];[1:v]scale=w=%d:h=%d[o2];[b][o2]overlay=%d:%d"\
             % (logo_mask_w, logo_mask_h, pos[0][2], pos[0][3], pos[0][0], pos[0][1], pos[1][2], pos[1][3], pos[1][0], pos[1][1])
    elif logo_num == 3:
        vf = "[0:v]scale=w=%d:h=%d[a];[1:v]scale=w=%d:h=%d[o];[a][o]overlay=%d:%d[b];" \
             "[1:v]scale=w=%d:h=%d[o2];[b][o2]overlay=%d:%d[c];[1:v]scale=w=%d:h=%d[o3];[c][o3]overlay=%d:%d" \
             % (logo_mask_w, logo_mask_h, pos[0][2], pos[0][3], pos[0][0], pos[0][1], pos[1][2], pos[1][3], pos[1][0], pos[1][1],
                pos[2][2], pos[2][3], pos[2][0], pos[2][1])
    elif logo_num >= 4:
        vf = "[0:v]scale=w=%d:h=%d[a];[1:v]scale=w=%d:h=%d[o];[a][o]overlay=%d:%d[b];" \
             "[1:v]scale=w=%d:h=%d[o2];[b][o2]overlay=%d:%d[c];[1:v]scale=w=%d:h=%d[o3];[c][o3]overlay=%d:%d[d];" \
             "[1:v]scale=w=%d:h=%d[o4];[d][o4]overlay=%d:%d" \
             % (logo_mask_w, logo_mask_h, pos[0][2], pos[0][3], pos[0][0], pos[0][1], pos[1][2], pos[1][3], pos[1][0], pos[1][1],
                pos[2][2], pos[2][3], pos[2][0], pos[2][1], pos[3][2], pos[3][3], pos[3][0], pos[3][1])

    else:
        # print '#input pos param incorrect, pls reset!'
        return None

    cmd_delogo = "%s -v quiet -i '%s' -i %s -filter_complex '%s' -f image2 -y '%s' " % (FFMPEG, black_mask, white_mask, vf, logo_mask_file)

    (ret, _) = commands.getstatusoutput(cmd_delogo)
    if ret:
        return None
    else:
        #return logo_mask_file
        if width == logo_mask_w and height == logo_mask_h:
            mask_logo = logo_mask_file
        elif os.path.isfile(logo_mask_file):
            if len(os.path.basename(logo_mask_file)) > 4 and '.' in os.path.basename(logo_mask_file):
                mask_logo = '%s_x.png' % logo_mask_file[:-4]  # .png
            else:
                mask_logo = '%s_x.png' % logo_mask_file  # os.path.join(video_out_dir, "%s_%dx%d_logo.png" % (basename0, width, height))
            cmd_logo = "%s -v quiet -i %s -s %dx%d -f image2 -y '%s' " % (FFMPEG, logo_mask_file, width, height, mask_logo)
            os.system(cmd_logo)
            if os.path.isfile(logo_mask_file):
                os.system("rm -f '%s' " % logo_mask_file)
        else:
            mask_logo = None

        return mask_logo


def get_ffmpeg_version(FFMPEG):
    ffmpeg_ver = FFMPEG_VERSION
    if os.path.isfile(FFMPEG):
        tmp_cmd = FFMPEG + " | grep 'ffmpeg version' "
        result = commands.getoutput(tmp_cmd)
        ffmpeg_version = result[result.find('ffmpeg version') + len('ffmpeg version'): result.find('Copyright')]
        ffmpeg_version = ffmpeg_version.replace(' ', '')
        ffv = ffmpeg_version[0:3]
        if '.' in ffv:
            ffmpeg_ver = float(ffv)
    else:
        print 'FFMPEG %s is not exist!' % FFMPEG
    return ffmpeg_ver


def get_dir_pics_list(tmp_dir, prefix='', suffix='.jpg', limit=0):
    pics_list = []
    for item in os.listdir(tmp_dir):
        fext = os.path.splitext(item)[1]
        if fext.lower() == suffix:
            filename = os.path.join(tmp_dir, item)
            if limit > 0:
                try:
                    rawsize = os.path.getsize(filename)  # byte
                    if rawsize < limit:
                        print '%s : %dB < %dB, del' % (filename, rawsize, limit)
                        # os.system("mv '%s' '%s'" % (filename, filename+'_.png'))
                        os.system("rm '%s'" % filename)
                        continue
                except:
                    pass
            if prefix and item[0:len(prefix)] == prefix:
                pics_list.append(filename)
            else:
                pics_list.append(filename)

    pics_list.sort()
    return pics_list


def get_dir_file_list(tmp_dir, prefix='', suffix='', extension='.mp4'):
    file_list = []
    for item in os.listdir(tmp_dir):
        fext = os.path.splitext(item)[1]
        if fext.lower() == extension:
            filename = os.path.join(tmp_dir, item)

            if prefix:
                if item[0:len(prefix)] == prefix:
                    file_list.append(filename)
            elif suffix:
                basename0 = os.path.basename(filename)
                basename0 = basename0[0: basename0.rfind('.')]
                if len(basename0) >= len(suffix) and basename0[len(suffix):-1] == prefix:
                    file_list.append(filename)
            else:
                file_list.append(filename)

    file_list.sort()
    return file_list


def ff_get_video_wxh(org_file):
    exe_str = "%s -hide_banner -i '%s' " % (FFMPEG, org_file)
    readline = commands.getoutput(exe_str)

    rr = size_pat.findall(readline)
    # print 'list wxh=', rr  # ['333x222', '1280x720']
    if rr and len(rr):
        try:
            ll = rr[-1].split('x')  # ['1280', '720']
            w, h = int(ll[0]), int(ll[1])
            if w > 0 and h > 0:
                return w, h
        except:
            pass

    return -1, -1


def get_video_bitrate(filename):
    bitrate = 0.0
    cmd_info_str = "%s -hide_banner -i '%s' " % (FFPROBE, filename)
    _, stdout = commands.getstatusoutput(cmd_info_str)
    br_pat = re.compile(r'bitrate:\s+([\d.-]+)\s+kb/s')  # bitrate: 294 kb/s

    n = br_pat.search(stdout)
    if n is not None:
        try:
            bitrates = float(n.group(1))
            if bitrates > 0.0:
                bitrate = bitrates
        except:
            pass

    if 0:
        br2_pat = re.compile(r'\s+([\d.-]+)\s+kb/s')  # 294 kb/s
        br_list = re.findall(br2_pat, stdout)
        vb = 0
        if len(br_list) >= 2:
            vb = float(br_list[1])
        if len(br_list) >= 3:
            ab = float(br_list[2])
            if vb < ab:  # usually vb > ab
                vb, ab = ab, vb

    return bitrate


# get video avg fps
def get_video_fps(filename):
    fps = 10.0
    cmd_info_str = "%s -hide_banner -i '%s' " % (FFPROBE, filename)
    _, stdout = commands.getstatusoutput(cmd_info_str)
    fps_pat = re.compile(r'\s+([\d.-]+)\s+fps')
    n = fps_pat.search(stdout)
    if n is not None:
        try:
            ffps = float(n.group(1))
            if 0 < ffps < 1000:
                fps = ffps  # vfps = int(ffps+0.5)
        except:
            pass

    return fps


def get_video_duration(filename):
    duration = 0.5

    # cmd_info_str = "%s -i '%s' -v quiet -select_streams v -show_streams " % (FFPROBE, filename) # jpg gif flv video== N/A
    cmd_info_str = "%s -i '%s' -v quiet -show_format " % (FFPROBE, filename)
    _, std_data = commands.getstatusoutput(cmd_info_str)

    duration_pat = re.compile(r'duration=\s?([\d.]+)')
    n = duration_pat.search(std_data)  # fixme format duration maybe not video stream duration
    if n:
        try:
            duration = float(n.group(1))
        except:
            print 'maybe picture, try guess duration = 0.5s!'
            duration = 0.5

    if duration < 1.0 and os.path.getsize(filename) > 100000:
        if 'format_name=avi' in std_data or 'format_name=flv' in std_data:
            new_name = filename + '_vv.mp4'
            vv_cmd = "%s -i '%s' -v quiet -an -c:v copy -f mp4 -y '%s'" % (FFMPEG, filename, new_name)
            ret, std_data2 = commands.getstatusoutput(vv_cmd)
            if ret == 0 and os.path.isfile(new_name):
                try:
                    filename = new_name
                    # TRY AGAIN
                    cmd_info_str = "%s -i '%s' -v quiet -show_format " % (FFPROBE, filename)
                    _, std_data = commands.getstatusoutput(cmd_info_str)
                    n = duration_pat.search(std_data)
                    if n:
                        duration = float(n.group(1))
                except:
                    duration = 0.5
                finally:
                    os.system("rm '%s'" % new_name)

    return duration


def get_video_duration_fps(filename):
    # 1. check file at least has video stream
    cmd_info_str = "%s -i '%s' -v quiet -show_format -show_streams" % (FFPROBE, filename)
    _, std_data = commands.getstatusoutput(cmd_info_str)

    video_pat = re.compile(r'codec_type=video')
    m = video_pat.search(std_data)
    if m is None:
        print 'the input file has no video stream'
        return -1, 10

    # 2. get fps and format duration
    fps = get_video_fps(filename)
    duration = get_video_duration(filename)

    return duration, fps


# 提取整个视频文件,视频分量,音频分量的所有时长和帧率,用于检测可能存在的问题
def ff_get_all_duration(filename, tmp_dir='tmp'):
    if not os.path.isfile(filename):
        return None
    if not os.path.isdir(tmp_dir):
        os.system("mkdir -p '%s'" % tmp_dir)
    # for some flv audio duration=N/A
    aa = os.path.join(tmp_dir, 'aa.mp4')
    vv = os.path.join(tmp_dir, 'vv.mp4')

    cmd_aa = "%s -v quiet -i '%s' -sn -vn -c:a copy -y '%s'" % (FFMPEG, filename, aa)
    os.system(cmd_aa)
    cmd_vv = "%s -v quiet -i '%s' -sn -an -c:v copy -y '%s'" % (FFMPEG, filename, vv)
    os.system(cmd_vv)

    info = dict()
    fps = get_video_fps(filename)
    duration = get_video_duration(filename)
    info['av'] = (duration, fps)
    info['audio'] = (0, 25)  # default
    info['video'] = (0, 25)

    if os.path.isfile(aa):
        try:
            a_fps = get_video_fps(aa)
            a_duration = get_video_duration(aa)
            info['audio'] = (a_duration, a_fps)
        except:
            pass
        finally:
            os.system("rm '%s'" % aa)

    if os.path.isfile(vv):
        try:
            v_fps = get_video_fps(vv)
            v_duration = get_video_duration(vv)
            info['video'] = (v_duration, v_fps)
        except:
            pass
        finally:
            os.system("rm '%s'" % vv)

    # print info
    return info


def ff_extract_audio(src_filename, dst_filename, suffix='.m4a', params=None):
    # params = {'copy': False, 'sound': True, 'ab': 64, 'ar': 44100, 'ac': 2, 'profile': 'aac_he', 'codec': 'libfdk_aac'}
    if not os.path.isfile(src_filename):
        raise Exception("src filename is not exist %s" % src_filename)

    filename, ext = os.path.splitext(dst_filename)
    out_file = filename + suffix

    if params:
        copy_mode = params.get('copy', False)
        sound_mode = params.get('sound', False)
        norm = params.get('norm', 0)
        denoise = 0
        volume = float(params.get('volume', 0.0))

        av_info = ffprobe_video_info(src_filename)
        # print av_info
        if not av_info or 'a_stream' not in av_info:
            raise Exception("src filename has no audio stream %s" % src_filename)

        if not suffix and 'avformat' in av_info:
            suffix = av_info['avformat'].get('format_name', '.m4a')
            out_file = filename + suffix

        if sound_mode:
            ab = int(params.get('ab', 64))  # 16
            ar = int(params.get('ar', 44100))  # 22050
            ac = int(params.get('ac', 2))
            profile = params.get('profile', '')
            if 'denoise' in params:
                denoise = int(params.get('denoise', denoise))
                if denoise == 2:
                    if av_info and 'a_stream' in av_info:
                        src_ar = float(av_info['a_stream'].get('sample_rate', 44100))
                        src_ab = float(av_info['a_stream'].get('bitrate', 64000))
                        if src_ar <= 16000 or src_ab <= 16000:
                            denoise = 1
        else:  # music_mode
            ab = int(params.get('ab', 64))  # 128
            ar = int(params.get('ar', 44100))
            ac = int(params.get('ac', 2))
            profile = params.get('profile', '')  # aac_he

        # valid audio param
        if ab < 8 or ab > 256:
            ab = 64
        if ar < 8000 or ar > 48000:
            ar = 44100
        if ac < 1 or ac > 2:
            ac = 2

        ap = ''
        if 22050 <= ar <= 48000 and profile:
            ap = '-profile:a %s' % profile

        af = ''
        if denoise:
            af = "-filter_complex highpass=f=200,lowpass=f=1600"  # 1200

        am = ''
        if norm:
            (mean_volume_2ch, max_volume_2ch) = detect_audio_volume(src_filename, aframes=1000)
            (mean_volume_1ch, max_volume_1ch) = detect_audio_volume(src_filename, aframes=1000, ac=1)
            # print mean_volume_2ch, max_volume_2ch
            # print mean_volume_1ch, max_volume_1ch

            if -99 < mean_volume_1ch < mean_volume_2ch - 5 or -99 < max_volume_1ch < -8:
                am = "-map_channel 0.%d.0" % (av_info['a_stream']['index'])

            if -30.0 < max_volume_2ch:
                if max_volume_2ch < -0.5:
                    if sound_mode:
                        add_volume = 5.0 - max_volume_2ch
                    else:
                        add_volume = 2.0 - max_volume_2ch
                else:
                    add_volume = 1.0 - max_volume_2ch

                add_volume += volume
                if abs(add_volume) > 0.3:
                    if af:
                        af += ",volume=%sdB" % add_volume
                    else:
                        af = "-af volume=%sdB" % add_volume

        audio_p = '-vn -sn -acodec libfdk_aac -ab %sk -ar %s -ac %s %s %s %s' % (ab, ar, ac, ap, af, am)

        if copy_mode:
            cmd_ex = "%s -v quiet -i '%s' -c:a copy -vn -sn -y '%s'" % (FFMPEG33, src_filename, out_file)
        else:
            cmd_ex = "%s -v quiet -i '%s' %s -y '%s'" % (FFMPEG33, src_filename, audio_p, out_file)

        # print cmd_ex
        ret, std_output = commands.getstatusoutput(cmd_ex)

    else:
        ret = 0
        std_output = "just copy src audio file!"
        out_file = src_filename

    if ret == 0 and os.path.isfile(out_file):
        av_info = ffprobe_video_info(out_file)
        if av_info and 'a_stream' in av_info:
            duration = float(av_info['a_stream'].get('duration', 0))
            bitrate = float(av_info['a_stream'].get('bit_rate', 0))  # bps
            if bitrate < 1.0:
                bitrate = float(av_info['avformat'].get('bit_rate', 0))
            format = av_info['avformat'].get('format_name', '')  # amr  mp3  mov,mp4,m4a
            if 'm4a' in format:
                format = 'm4a'

            if format.upper() not in support_list:
                print "format unknown, just set to m4a!"
                format = 'm4a'

            return {'output': out_file, 'duration': duration, 'bitrate': bitrate, 'format': format}

    print std_output
    raise Exception("can't transcode audio %s" % std_output)


def audio_silence_detect(fname):
    silence_pat=re.compile(r'silence_end:\s?([\d.-]+)\s?| silence_duration:\s?([\d.-]+)')  # silence_end: 384.692 | silence_duration: 162.269
    cmd = "%s -hide_banner -i '%s' -af silencedetect=n=-55dB:d=5 -vn -f null -" % (FFMPEG33, fname)
    # print cmd
    _, stdout = commands.getstatusoutput(cmd)
    xx = silence_pat.findall(stdout)  # yy = re.findall(silence_pat, stdout)

    err_msg = ''
    data = []
    if len(xx):
        # print stdout
        err_msg += 'audio has silence'
        for i in range(0, len(xx)/2):
            se = xx[i*2][0]
            sd = xx[i*2+1][1]
            try:
                fend = float(se)
                if fend > 8.0:  # for some mtv pre start 6s is silence!
                    data.append((se, sd))  # [(silence_end, silence_duration),]
            except:
                pass

    return err_msg, data


def check_audio_channel_volume_silence(stderr_info):
    if not stderr_info:
        return 1

    mean_volume_1ch = No_volume_value
    max_volume_1ch = No_volume_value
    n = mean_volume_pat.search(stderr_info)
    m = max_volume_pat.search(stderr_info)
    if n is not None:
        try:
            mean_volume_1ch = float(n.group(1))
        except:
            pass
    if m is not None:
        try:
            max_volume_1ch = float(m.group(1))
        except:
            pass

    if No_volume_value < mean_volume_1ch < -36 and No_volume_value < max_volume_1ch < -30:
        return 1
    else:
        return 0


def select_audio_channel(fname, stream_id=1, dir='./output'):
    flag = 0
    ret_ch = 0
    basename0 = os.path.basename(fname)
    output_mp4_tmp = os.path.join(dir, "%s_audio_1ch_xx.mp4" % basename0)
    cmd_info_str = "%s -v quiet -i '%s' -vn -sn -aframes 200  -map_channel 0.%d.0 -y '%s' " % (FFMPEG, fname, stream_id, output_mp4_tmp)
    # print cmd_info_str
    try:
        (r, _) = commands.getstatusoutput(cmd_info_str)
        if r == 0 and os.path.isfile(output_mp4_tmp):
            cmd_detect = "%s -i '%s' -af volumedetect -f mp4 -y /dev/null " % (FFMPEG, output_mp4_tmp)
            try:
                (ret, stderr_info) = commands.getstatusoutput(cmd_detect)
                if check_audio_channel_volume_silence(stderr_info):
                    print 'the first audio channel is silence'
                    flag = 1
                    ret_ch = 1
            except:
                pass
    except:
        flag = 1

    if flag == 1:
        cmd_info_str = "%s -v quiet -i '%s' -vn -sn -aframes 200  -map_channel 0.%d.1 -y '%s' " % (FFMPEG, fname, stream_id, output_mp4_tmp)
        # print cmd_info_str
        try:
            (r, _) = commands.getstatusoutput(cmd_info_str)
            if r == 0 and os.path.isfile(output_mp4_tmp):
                cmd_detect = "%s -i '%s' -af volumedetect -f mp4 -y /dev/null " % (FFMPEG, output_mp4_tmp)
                try:
                    (ret, stderr_info) = commands.getstatusoutput(cmd_detect)
                    if check_audio_channel_volume_silence(stderr_info):
                        print 'the second audio channel is silence'
                        ret_ch = 0
                    else:
                        ret_ch = 1
                except:
                    pass
            else:  # no channal_1
                ret_ch = 0
        except:
            ret_ch = 0
            pass

    if os.path.isfile(output_mp4_tmp):
        os.system("rm -f '%s' " % output_mp4_tmp)

    return ret_ch


def detect_audio_volume(fname, aframes=0, ac=2, dir='./output'):
    """
    ac=1 we got the 1ch volume
    [Parsed_volumedetect_0 @ 0x7f852b416280] mean_volume: -15.2 dB
    [Parsed_volumedetect_0 @ 0x7f852b416280] max_volume: 0.0 dB
    """
    if not os.path.isfile(fname):
        print fname, 'is not exist!'
        return

    if dir and not os.path.isdir(dir):
        os.system("mkdir -p '%s'" % dir)

    basename0 = os.path.basename(fname)
    output_mp4_tmp = os.path.join(dir, "%s_try_1ch_tmp.mp4" % basename0)

    mean_volume = No_volume_value
    max_volume = No_volume_value

    delete_tmp = 0
    audio_channel = ' -ac %d ' % ac

    if aframes > 0:
        aframes = ' -aframes %d ' % aframes
    else:
        aframes = ''

    if ac == 1:
        if os.path.isfile(FFMPEG2):
            cmd_info_str = "%s -i '%s' -af volumedetect %s %s -vn -sn -f null -y /dev/null" % (FFMPEG2, fname, audio_channel, aframes)
        else:
            ffmpeg_ver = get_ffmpeg_version(FFMPEG)

            # for ffmpeg3.0 it need to creat a 1ch video to detect 1ch audio
            if ffmpeg_ver > 2.8 and not os.path.isfile(FFMPEG2):
                ffmpeg_video_params = ' -sn -c:v libx264 -preset ultrafast -s 176x144'
                ffmpeg_audio_params = ' -af volumedetect -ac 1'  # should use -af volumedetect get 2000frames of new input file

                cmd_info_str = "%s -v quiet -i '%s' -vframes 500  %s %s -f mp4 -y '%s' " % (
                    FFMPEG_FIX, fname, ffmpeg_audio_params, ffmpeg_video_params, output_mp4_tmp)

                (_, stderr_data_tmp) = commands.getstatusoutput(cmd_info_str)

                if os.path.isfile(output_mp4_tmp):
                    delete_tmp = 1
                    cmd_info_str = "%s -i '%s' -af volumedetect -vn -sn -f null -y /dev/null" % (FFMPEG_FIX, output_mp4_tmp)
                else:
                    return mean_volume, max_volume
            else:  # FFMPEG = 2.6.1
                cmd_info_str = "%s -i '%s' -af volumedetect %s %s -vn -sn -f null -y /dev/null" % (FFMPEG, fname, audio_channel, aframes)
    else:
        cmd_info_str = "%s -i '%s' -af volumedetect %s %s -vn -sn -f null -y /dev/null" % (FFMPEG_FIX, fname, audio_channel, aframes)

    (_, stderr_data) = commands.getstatusoutput(cmd_info_str)

    if delete_tmp == 1 and os.path.isfile(output_mp4_tmp):
        os.system("rm -f '%s' " % output_mp4_tmp)

    n = mean_volume_pat.search(stderr_data)
    m = max_volume_pat.search(stderr_data)
    if n is not None:
        try:
            mean_volume = float(n.group(1))  # re第一个括弧内匹配的内容
        except:
            pass
    if m is not None:
        try:
            max_volume = float(m.group(1))
        except:
            pass

    return mean_volume, max_volume


def normalize_audio_file(input_filename, output_filename, av_out_info, dst_dir):
    (mean_volume_2ch, max_volume_2ch) = detect_audio_volume(input_filename, aframes=1000)
    (mean_volume_1ch, max_volume_1ch) = detect_audio_volume(input_filename, aframes=1000, ac=1, dir=dst_dir)

    ret = 0
    ffmpeg_audio_params = ' -acodec libfdk_aac -ab 128k -ar 44100 -ac 2 -profile:a aac_he '
    if -99 < mean_volume_1ch < mean_volume_2ch - MEAN_VOLUME_DIFF_DTHRES or -99 < max_volume_1ch < MAX_VOLUME_1CH_THRES:  # if mix to ac=1 the audio is much small than channel 1
        ffmpeg_audio_params += ' -map_channel 0.%d.0 ' % (av_out_info['a_stream']['index'])
        av_out_info['err_msg'] += '!2ch->1ch'
        ret = 2

    if -30 < max_volume_2ch < MAX_VOLUME_THRES/2:  # MAX_VOLUME_THRES:  # if max_volume<-30 it maybe just background noise, no mean to improve
        add_volume = -2.0 - max_volume_2ch
        ffmpeg_audio_params += ' -af volume=%sdB ' % add_volume
        av_out_info['err_msg'] += '!audio volume'
        ret = 3

    if ret > 0:
        ffmpeg_video_params = ' -c:v copy -sn '
        cmd_str_wav = "%s -v quiet -i '%s' %s %s -f mp4 -y '%s' " % (FFMPEG33, input_filename, ffmpeg_video_params, ffmpeg_audio_params, output_filename)
        print 'try fix audio problem: %s' % cmd_str_wav
        ret_wav = os.system(cmd_str_wav)
        if ret_wav:
            print 'input audio file may has problem'
            return -1

    return ret


def ffmpeg_cropdetect(input_filename):
    # ./ffmpeg -i /Users/jqhe/error_1.mp4 -vf cropdetect -vframes 2 -an -sn -f mp4 -y /dev/null
    cmd_crop_str = "%s -i '%s' -vf cropdetect -vframes 4 -an -sn -f mp4 -y /dev/null" % (FFMPEG, input_filename)
    (_, result_crop_info) = commands.getstatusoutput(cmd_crop_str)

    searched_all = re.findall('crop=(\d+):(\d+):(\d+):(\d+)', result_crop_info)
    # print searched_all
    if searched_all:
        crop = searched_all[-1]
        crop_data = list(crop)
        # print crop_data
    else:
        return None

    '''
    searched = re.search('crop=(\d+):(\d+):(\d+):(\d+)', result_crop_info)
    if searched:
        try:
            crop = searched.group()  # crop=1920:832:0:124 [w:h:x:y]
        except:
            return None
    else:
        return None

    crop = crop[crop.find('=') + 1:]
    crop_data = crop.strip().split(':')
    '''

    if len(crop_data) == 4:
        if int(crop_data[2]) > 4 or int(crop_data[3]) > 4:
            print 'crop: %s' % str(crop)
        return crop_data
    else:
        return None


# ffprobe err for quicktime stop play force -r fps
def ffprobe_video_err(input_filename):
    if os.path.isfile(input_filename):
        cmd_info_str = "%s -print_format json -show_format -show_streams -i '%s' " % (FFPROBE_FIX, input_filename)

        _, stderr_data = commands.getstatusoutput(cmd_info_str)
        # print _, stderr_data

        if 'AV_TIME_BASE, Reset duration to 0' in stderr_data:
            # print '#info: AV_TIME_BASE, Reset duration to 0'
            return 1
    return 0


def ffprobe_video_info(input_filename, ffprobe=FFPROBE33):
    if os.path.isfile(input_filename):
        av_out_info = dict()
        av_out_info['err_code'] = 0
        av_out_info['err_msg'] = ''
        stderr_data = ''

        if os.path.isfile(ffprobe) or os.path.isfile(FFPROBE_FIX):
            cmd_info_str = "%s -v quiet -print_format json -show_format -show_streams -i '%s' " % (ffprobe, input_filename)
            (ret, stderr_data) = commands.getstatusoutput(cmd_info_str)
            if ret:
                if os.path.isfile(FFPROBE_FIX):
                    print '#ffprobe has err, use ffprobe_fix'
                    av_out_info['err_msg'] = 'ffprobe ok fix'
                    cmd_info_str = "%s -v quiet -print_format json -show_format -show_streams -i '%s' " % (FFPROBE_FIX, input_filename)
                    (ret, stderr_data) = commands.getstatusoutput(cmd_info_str)
        else:
            print "ffprobe: %s is not exist!" % ffprobe
            return None

        json_str = stderr_data
        try:
            dict_info = json.loads(json_str, encoding='utf-8-sig')  # it is better for chinese character? or just use 'utf-8'
        except:
            return None

        avformat = dict()
        if 'format' in dict_info:
            avformat.update(dict_info['format'])
        else:
            return None

        if 'tags' in avformat:
            if 'spherical' in avformat['tags']:
                sp = avformat['tags']['spherical']
                if sp != ' ' and sp != '0' and sp != 'false' and sp != 'null':
                    avformat['spherical'] = sp
            del (avformat['tags'])
        if 'nb_programs' in avformat:
            del (avformat['nb_programs'])
        if 'probe_score' in avformat:
            del (avformat['probe_score'])
        if 'format_long_name' in avformat:
            del (avformat['format_long_name'])
        # print avformat

        if avformat and avformat['nb_streams'] > 0:
            av_out_info['avformat'] = copy.deepcopy(avformat)
        else:
            av_out_info['err_code'] = -1
            av_out_info['err_msg'] = 'ffprobe no streams'
            return None

        list_stream = dict_info['streams']
        # print type(list_stream), len(list_stream),'\n', list_stream

        a_stream = {}
        v_stream = {}
        # s_stream = {} FIXME subtitile

        a_stream_count = 0
        v_stream_count = 0
        # FIXME maybe defined a common parameter sets, but still not sure which paras is nessary
        for stream in list_stream:
            if isinstance(stream, dict):
                if 'codec_type' in stream and stream['codec_type'] == 'audio':
                    a_stream.update(stream)
                    if 'disposition' in a_stream:
                        del (a_stream['disposition'])
                    if 'codec_tag_string' in a_stream:
                        del (a_stream['codec_tag_string'])
                    if 'codec_tag' in a_stream:
                        del (a_stream['codec_tag'])
                    if 'codec_time_base' in a_stream:
                        del (a_stream['codec_time_base'])
                    if 'dmix_mode' in a_stream:
                        del (a_stream['dmix_mode'])
                    if 'ltrt_cmixlev' in a_stream:
                        del (a_stream['ltrt_cmixlev'])
                    if 'ltrt_surmixlev' in a_stream:
                        del (a_stream['ltrt_surmixlev'])
                    if 'loro_cmixlev' in a_stream:
                        del (a_stream['loro_cmixlev'])
                    if 'loro_surmixlev' in a_stream:
                        del (a_stream['loro_surmixlev'])
                    if 'side_data_list' in a_stream:
                        del (a_stream['side_data_list'])
                    if 'tags' in a_stream:
                        del (a_stream['tags'])

                    a_stream_count += 1
                    if a_stream_count == 1:
                        av_out_info['a_stream'] = copy.deepcopy(a_stream)
                    else:
                        a_key = 'a_stream_%d' % a_stream_count
                        av_out_info[a_key] = copy.deepcopy(a_stream)

                if 'codec_type' in stream and stream['codec_type'] == 'video':
                    v_stream.update(stream)
                    if 'disposition' in v_stream:
                        if 'attached_pic' in v_stream['disposition']:
                            v_stream['attached_pic'] = v_stream['disposition']['attached_pic']
                        del (v_stream['disposition'])
                    if 'codec_tag_string' in v_stream:
                        del (v_stream['codec_tag_string'])
                    if 'codec_tag' in v_stream:
                        del (v_stream['codec_tag'])
                    if 'codec_time_base' in v_stream:
                        del (v_stream['codec_time_base'])
                    if 'tags' in v_stream:
                        if 'rotate' in v_stream['tags']:
                            v_stream['rotate'] = v_stream['tags']['rotate']  # 'rotate' = '90' means when display 'rotation' = -90
                        del (v_stream['tags'])

                    v_stream_count += 1
                    if v_stream_count == 1:
                        av_out_info['v_stream'] = copy.deepcopy(v_stream)
                    else:
                        v_key = 'v_stream_%d' % v_stream_count
                        av_out_info[v_key] = copy.deepcopy(v_stream)

        av_out_info['a_nb_streams'] = a_stream_count
        av_out_info['v_nb_streams'] = v_stream_count

        if a_stream_count > 1:
            if int(av_out_info['a_stream']['index']) > int(av_out_info['a_stream_2']['index']):
                av_out_info['a_stream'], av_out_info['a_stream_2'] = av_out_info['a_stream_2'], av_out_info['a_stream']  # switch
                # print av_out_info['a_stream'], av_out_info['a_stream_2']

        if v_stream_count > 1:
            if int(av_out_info['v_stream']['index']) > int(av_out_info['v_stream_2']['index']):
                av_out_info['v_stream'], av_out_info['v_stream_2'] = av_out_info['v_stream_2'], av_out_info['v_stream']

        if 0 and os.path.isfile(FFMPEG):  # just for debug
            tmp_cmd = FFMPEG + " | grep 'ffmpeg version' "
            result = commands.getoutput(tmp_cmd)
            ffmpeg_version = result[result.find('ffmpeg version') + len('ffmpeg version'): result.find('Copyright')]
            ffmpeg_version = ffmpeg_version.replace(' ', '')
            av_out_info['ffmpeg_version'] = ffmpeg_version

        av_out_info['err_msg'] = 'ffprobe ok!'
        return av_out_info


def guess_crf_from_psnr(stderr_info, w_in=360, w_out=360):
    # w_in: input min(w,h)
    # w_out: output min(w,h)
    crf = CRF_DEFAULT

    if stderr_info:
        mean_psnr1 = None
        bt1 = None
        mean_psnr = ''

        psnr_file = None  # psnr_file = open('psnr_cmp.log', 'a+')

        if os.path.isfile(stderr_info):
            bt_cmd = "cat -v '%s' |grep 'kb/s:'| tail -1" % stderr_info
            readline = os.popen(bt_cmd)
            bt_line = readline.read().strip()
            if bt_line:
                bt1 = re.findall('kb/s:([0-9]+\.?[0-9]+)', bt_line)

            psnr_cmd = "cat -v '%s' |grep 'PSNR Mean Y:'| tail -1" % stderr_info
            readline = os.popen(psnr_cmd)
            psnr_line = readline.read().strip()
            # print psnr_line
            if psnr_line:
                mean_psnr1 = re.findall('Y:([0-9]+\.?[0-9]+) U:([0-9]+\.?[0-9]+) V:([0-9]+\.?[0-9]+)', psnr_line)  # [('41.537', '47.168', '47.451')]
        else:
            # print stderr_info
            if psnr_file:
                psnr_file.write('---------------\n')

            bt1 = re.findall('kb/s:([0-9]+\.?[0-9]+)', stderr_info)  # [300.0]

            mean_psnr1 = re.findall('PSNR Mean Y:([0-9]+\.?[0-9]+) U:([0-9]+\.?[0-9]+) V:([0-9]+\.?[0-9]+)', stderr_info)

        if mean_psnr1:
            mean_psnr = mean_psnr1[-1]

        if bt1:
            try:
                br = float(bt1[0])
                if 0 < br < 4:
                    return CRF_DEFAULT - 2
                elif 4 <= br < 8:
                    return CRF_DEFAULT - 1

                if br > 500:  # FIXME should increase the maxbitrate, crf has no use; or deshake deblock denoise...
                    return CRF_DEFAULT
            except:
                pass
        ##############################
        print 'PSNR Mean: %s' % str(mean_psnr)
        if psnr_file:
            psnr_file.write('PSNR Mean: %s\n' % str(mean_psnr))
            psnr_file.close()

        try:
            mean_psnr_y = float(mean_psnr[0])
            mean_psnr_u = float(mean_psnr[1])
        except:
            mean_psnr_y = 0.0
            mean_psnr_u = 0.0

        if w_out/(w_in+1.0) > 1.8:
            p_offset = 1.5
        elif w_out/(w_in+1.0) > 1.3:
            p_offset = 1.0
        else:
            p_offset = 0.5

        if mean_psnr_y > 1.0:
            if mean_psnr_y > 47 + p_offset:
                crf += 1.0
            elif mean_psnr_y > 45 + p_offset:
                crf += 0.5
            elif mean_psnr_y < 36.0 - p_offset and mean_psnr_u < 43.0:  # FIXME 可能需要去抖动 或者 强自适应平滑滤波 或者增大最大码率参数
                crf -= 0.5

    return crf


# 在python的函数形参中 string tuple number是不可改变的对象，只是标签引用；list dict是可以修改的对象
def get_video_sar(av_out_info):
    # 这个参数主要在concat两个视频时,如果sar不一致, wxh无法作为第一个视频的参考放缩大小,而只能用dar对应的w1xh
    sar = u'1/1'
    if av_out_info is None:
        return None

    if 'sample_aspect_ratio' in av_out_info['v_stream']:
        u_sar = av_out_info['v_stream']['sample_aspect_ratio']
        av_out_info['sar'] = u_sar  # FIXME should add sar anywhere?
        if u_sar == '1/1' or u_sar == '1:1':
            pass
        else:
            if ':' in u_sar or '/' in u_sar:
                if ':' in u_sar:
                    [u_sar_w, u_sar_h] = u_sar.split(':')
                else:
                    if u_sar == 'N/A':  # needed?
                        u_sar = u'0:1'
                        av_out_info['sar'] = u_sar
                        return sar
                    else:
                        [u_sar_w, u_sar_h] = u_sar.split('/')

                sar_w = int(u_sar_w.strip())
                sar_h = int(u_sar_h.strip())
                if sar_w > 0 and sar_h > 0:
                    sar_new = float(sar_w) / float(sar_h)
                    if abs(sar_new - 1.0) < 0.01:  # 0.01 almost the same like 128:117= 1.094
                        pass
                    else:
                        sar = u'%d/%d' % (sar_w, sar_h)
                        # print 'sar=', u_sar
                        av_out_info['sar'] = u_sar  # sar=4/3  u_sar=4:3
                        if 'display_aspect_ratio' in av_out_info['v_stream']:
                            av_out_info['dar'] = av_out_info['v_stream']['display_aspect_ratio']
                            if 'height' in av_out_info['v_stream'] and 'width' in av_out_info['v_stream']:
                                av_out_info['disp_height'] = int(av_out_info['v_stream']['height'])
                                if isinstance(av_out_info['v_stream']['width'], int):
                                    disp_width = av_out_info['v_stream']['width']
                                else:
                                    disp_width = int(str(av_out_info['v_stream']['width']).strip())
                                disp_width = disp_width * sar_w / sar_h
                                disp_width = (disp_width + 6) / 8 * 8  # mod 8
                                av_out_info['disp_width'] = disp_width
                                # .ffmpeg -i $f0 -vf "scale=w=iw*sar:h=ih,setsar=r=1/1" -c:a copy -y out.mp4
                else:  # like 0:1
                    if 'height' in av_out_info['v_stream'] and 'width' in av_out_info['v_stream']:
                        # if sar_w > 0 or sar_h > 0:
                        av_out_info['disp_width'] = int(av_out_info['v_stream']['width'])
                        av_out_info['disp_height'] = int(av_out_info['v_stream']['height'])
                    return '%d/%d' % (sar_w, sar_h)
    return sar


# get float video fps from av_out_info
def get_video_vfps(av_out_info):
    vfps = 20
    if 'r_frame_rate' in av_out_info['v_stream'] and av_out_info['v_stream']['r_frame_rate'] != 'N/A':
        u_fps = av_out_info['v_stream']['r_frame_rate'].strip()
    elif 'avg_frame_rate' in av_out_info['v_stream'] and av_out_info['v_stream']['avg_frame_rate'] != 'N/A':
        u_fps = av_out_info['v_stream']['avg_frame_rate'].strip()
    else:
        return vfps  # None

    vfps_d = '20'
    vfps_n = '1'
    if ':' in u_fps or '/' in u_fps:
        if ':' in u_fps:
            [vfps_n, vfps_d] = u_fps.split(':')
        else:
            [vfps_n, vfps_d] = u_fps.split('/')

    vfps_dd = int(vfps_d.strip())
    if vfps_dd > 0:
        vfps = float(vfps_n.strip()) / vfps_dd  # vfps = (int(vfps_n.strip()) + vfps_dd / 2) / vfps_dd

    if vfps > 30:
        # fixme it means not constant fps!
        if 'filename' in av_out_info['avformat']:
            cmd_info_str = "%s -i '%s' " % (FFPROBE, av_out_info['avformat']['filename'])
            (ret, stdout) = commands.getstatusoutput(cmd_info_str)

            fps_pat = re.compile(r'\s+([\d.-]+)\s+fps')
            n = fps_pat.search(stdout)
            if n is not None:
                try:
                    ffps = float(n.group(1))
                    if ffps >= 1:
                        vfps = ffps  # vfps = int(ffps+0.5)
                    else:
                        vfps = 10  # maybe gif ppt
                    print 'real fps -> %s' % vfps
                except:
                    pass

    return vfps


def get_av_duration_from_info(av_out_info):
    a_duration = 0.0
    v_duration = 0.0
    if 'v_stream' in av_out_info and 'duration' in av_out_info['v_stream']:
        try:
            v_duration = float(av_out_info['v_stream']['duration'])
        except:
            # print 'no v_duration'
            pass
    if 'a_stream' in av_out_info and 'duration' in av_out_info['a_stream']:
        try:
            a_duration = float(av_out_info['a_stream']['duration'])
        except:
            # print 'no a_duration'
            pass
    return [v_duration, a_duration]


##########################
def ff_get_video_rotate(input_file='', info=''):
    if not info:
        exe_str = "%s -select_streams v -show_streams -v quiet -i '%s' " % (FFPROBE33, input_file)
        stderr_info = commands.getoutput(exe_str)
        info = stderr_info
    # print info
    rotate_pat = re.compile(r'rotate=\s?([\d.-]+)')
    rotate = 0
    if 'rotate' in info:
        r = rotate_pat.search(info)
        if r is not None:
            try:
                rotate = int(r.group(1).strip())
            except:
                pass

    return rotate


def ff_get_video_sar(input_file='', info=''):
    if not info:
        ff_cmd = "%s -hide_banner -i '%s'" % (FFMPEG, input_file)
        info = commands.getoutput(ff_cmd)
    sar_pat = re.compile(r'SAR\s+(\d+:\d+)')  # SAR 128:127
    # print info
    if 'SAR' in info:
        s = sar_pat.search(info)
        if s is not None:
            try:
                sar_x = s.group(1).strip()  # get the first () matched content
                x, y = sar_x.split(':')
                x, y = int(x), int(y)
                return x, y  # '%s/%s' % (x, y)
            except:
                pass
    return 1, 1


def ff_get_video_display_wxh(input_file, allow_rotate=1, allow_sar=0):
    exe_str = "%s -hide_banner -i '%s' " % (FFMPEG, input_file)
    info = commands.getoutput(exe_str)
    info = info[info.find('Stream'):]
    rr = size_pat.findall(info)
    # print 'list wxh=', rr  # ['333x222', '1280x720']
    if rr and len(rr):
        try:
            ll = rr[-1].split('x')  # ['1280', '720']
            w, h = int(ll[0]), int(ll[1])
            if len(rr) > 1:
                ll2 = rr[0].split('x')  # ['1280', '720']
                w2, h2 = int(ll2[0]), int(ll2[1])
                if w2*h2 > w*h:
                    w, h = w2, h2
            # print "src_raw wxh = %sx%s" % (w, h)
            if w > 0 and h > 0:
                if allow_rotate:
                    rotate = ff_get_video_rotate(input_file)
                    if (rotate / 90) % 2:  # rotate % 180:
                        w, h = h, w

                if allow_sar:
                    x, y = ff_get_video_sar(info=info)
                    if x > 0 and y > 0:
                        if x > y:
                            w = w*x/y
                            w += (w % 2)
                        elif x < y:
                            h = h*y/x
                            h += (h % 2)
                # print "src_fix wxh = %sx%s" % (w, h)
                return w, h
        except:
            pass

    return -1, -1


def main():
    # info = ff_get_all_duration('/data/39c9a4d88fec4b6d924694c428efd8e4.mp4')

    err_msg, data = audio_silence_detect('/data/39c9a4d88fec4b6d924694c428efd8e4.mp4')  # '/data/39c9a4d88fec4b6d924694c428efd8e4.mp4'

    d, fps = get_video_duration_fps(r'/data/last_frames_err1.mp4')  # (r'/data/Compare_1114/a1.mp4')
    # av_interleave('/data/dst_audio_wav.mp4')  # '/data/dst_audio_wav' '/data/Compare_1114/bk/v1.mp4' '/data/videos/audio_err/35691.mp4'
    param = {'h': 360, 'pos': [[42, 29, 81, 30], [483, 31, 81, 14]], 'w': 640, 'logo_mask_file': 'logo_mask_file.jpg'}
    r = create_logo_mask(param, 640, 360)

    # input_filename = r'/data/videos/2050517774.mp4'
    input_filename = r'/data/videos/6103692770.mp4'

    # av_out_info = ffprobe_video_info(input_filename)

    av_out_info = ffprobe_video_err(input_filename)

    print av_out_info

if __name__ == '__main__':
    # sar = ff_get_video_sar('/data/videos/scale-org-360p.mp4')
    # main()
    print ffprobe_video_info("../test/test.mp4")

    import time
    star_time = time.time()
    info = ffprobe_video_info('/data/videos/6103692770.mp4')
    print "time: %s" % str(time.time() - star_time)
    print info

