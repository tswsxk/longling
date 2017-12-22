'''
@author:hejianqiang
'''
#!/usr/bin/python -w
# -*- coding:utf8 -*-

import commands
import os
import re

from longling.framework.keyScene.common import FFMPEG, FFMPEG33, FFMPEG_PGC, sufset, get_video_duration_fps, \
    get_dir_pics_list, ff_get_first_pic, ff_get_last_pic

VERSION = '2017-2-27'

size_pat = re.compile(r'[0-9]{2,4}x[0-9]{2,4}')  # 640x360 # {2~4} num of digital x {2~4} num of digital
rotate_pat = re.compile(r'rotate\s+:\s?([\d.-]+)')  # rotate   : 270


def ff_check_rotate(org_file):
    exe_str = "%s -i '%s' " % (FFMPEG, org_file)
    stderr_info = commands.getoutput(exe_str)
    info = stderr_info[stderr_info.find('Metadata:'):]
    # print info
    rotate = 0
    if 'rotate ' in info:
        r = rotate_pat.search(info)
        if r is not None:
            try:
                rotate = int(r.group(1).strip())
            except:
                pass

    return rotate


# max_w = -1, max_h = -1 will return w, h
def ff_get_pic_wxh_360p(org_file, max_w=640, max_h=360, allow_rotate=0):
    exe_str = "%s -hide_banner -i '%s' " % (FFMPEG, org_file)
    readline = commands.getoutput(exe_str)
    if 'Video:' not in readline:
        raw_filesize = os.path.getsize(org_file)
        print '%s has not video element! filesize:%s' % (org_file, raw_filesize)
        return 0, 0

    rr = size_pat.findall(readline)
    # print 'list wxh=', rr  # ['333x222', '1280x720']
    if rr and len(rr):
        try:
            ll = rr[-1].split('x')  # ['1280', '720']
            w, h = int(ll[0]), int(ll[1])
            if w > 0 and h > 0:
                rotate = ff_check_rotate(org_file)
                if (rotate / 90) % 2:
                    w, h = h, w

                if max_w < 16 or max_h < 16:
                    return w, h

                if allow_rotate and w < h:  # for ver video 360x640
                    max_w, max_h = max_h, max_w

                if w/h > max_w/max_h:
                    width = max_w
                    height = width * h/w
                else:
                    height = max_h
                    width = height * w/h

                width += (width % 2)
                height += (height % 2)
                return width, height
        except:
            pass

    return max_w, max_h


def ff_get_pictures_360p(filename, tmp_dir='tmp', capture_num=20, fps=0.2, vf=None, width=-1, height=-1, ss=0.0, limit=0):
    if width < 16 and height < 16:
        width, height = ff_get_pic_wxh_360p(filename)  # 640, 360

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

    outfile = os.path.join(tmp_dir, r'%03d.jpg')  # 001~999

    cmd_str = "%s -v quiet %s -i '%s' -sn -an -f image2 -q:v 2 -vframes %d -vf 'fps=fps=%.2f%s' -pix_fmt yuvj420p -y '%s' " \
              % (FFMPEG33, start_time, filename, int(capture_num), fps, filters, outfile)
    ret, _ = commands.getstatusoutput(cmd_str)
    pic_list = get_dir_pics_list(tmp_dir, prefix='', suffix='.jpg', limit=limit)
    return pic_list


def yuv2jpg(filename, vf=None, yuvsize=None, output='', w=-1, h=-1):
    if not os.path.isfile(filename):
        # print 'input file is not exist!'
        return None

    if vf is None:
        vf = {}
    if yuvsize is None:
        yuvsize = ''

    basename0 = os.path.basename(filename)  # p_5_640x360.yuv
    filepath0 = os.path.split(filename)[0]

    if '_' in basename0:
        basename1 = basename0[:basename0.rfind('_')]  # rm p_ _640x360
    else:
        basename1 = basename0[:]

    if 'p_' in basename1:
        basename1 = basename1.replace('p_', '')
    if len(basename1) == 0:
        basename1 = '0'

    if not output:
        output = os.path.join(filepath0, basename1 + '.jpg')

    size_in = None
    if yuvsize:
        # print 'input yuv size:%s' % yuvsize
        x = size_pat.search(yuvsize)
        if x is not None:
            size_in = yuvsize

    if size_in is None:
        x = size_pat.search(basename0.replace(' ', ''))
        if x is not None:
            size_in = x.group()
        else:
            # print 'can not get yuv size!'
            return None

    # print size_in
    if w > 16 or h > 16:
        if w > 16 and h > 16:  # FIXME output wxh should be even, but w h != -1
            w = (w + 1) / 2 * 2
            h = (h + 1) / 2 * 2
        if vf and 'vf' in vf:
            filters = '-vf scale=%d:%d,%s' % (w, h, vf['vf'])
        else:
            filters = '-vf scale=%d:%d' % (w, h)
    else:
        if vf and 'vf' in vf:
            filters = '-vf %s' % (vf['vf'])
        else:
            filters = ''

    ff_cmd = "%s -pix_fmt yuv420p -s %s -i '%s' %s -f image2 -q:v 2 -pix_fmt yuvj420p -y '%s' " % (FFMPEG, size_in, filename, filters, output)
    # print ff_cmd
    ret, _ = commands.getstatusoutput(ff_cmd)

    if os.path.isfile(filename):  # and ret == 0:
        os.system("rm -f '%s' " % filename)  # rm input yuv file
    else:
        pass # print _

    if os.path.isfile(output):
        try:
            idx = int(basename1)
            return idx  # output
        except:
            pass

    # print '-----------yuv2jpg occur error: %s' % filename
    return None


def is_poc_inlist(plist, poc, r=0):
    for i in range(poc-r, poc+r+1):
        if i in plist:
            return True
    return False


'''
input: @param {}
must include parameters:
    param['input'] = '/data/videos/x.mp4' # the input video file
    param["tmp_dir"] = 'tmpx' # for store the output pictures

addtional input parameters[not needed if just use default]:
    param["width"] = 640  # output pic wxh
    param["height"] = 360
    param["vf"] = 'extend_range' # special filter for output pictures
==================
output: { 'err_code': 0, 'err_msg': 'ok!', 'duration': 134.68,  'capture_num': 19,
        'capture_frames': [2890, 3102, 327, 5, 224, 448, 673, 897, 1122, ...]}
'''

def get_scenecut_frame(param):
    tmp_dir = 'tmp'
    width = -1
    height = -1

    if 'input' in param and os.path.isfile(param['input']):
        input_filename = param['input']
    else:
        return {'err_code': -2, 'err_msg': 'file not exist!'}

    if 'width' in param:
        width = param["width"]

    if 'height' in param:
        height = param["height"]

    if width < 16 or height < 16:
        width, height = ff_get_pic_wxh_360p(input_filename)
        if not width or not height:
            return {'err_code': -3, 'err_msg': 'seems not a video!'}

    if 'vf' in param and param['vf']:
        vf_filter = {'vf': param['vf']}
    else:
        vf_filter = None

    if 'tmp_dir' in param:
        tmp_dir = param["tmp_dir"]

    # TODO... max frame num: only for hotsoon aweme etc < 16s videos currently
    try:
        max_fn = int(param.get('max_frames', 0))
    except:
        max_fn = 0

    # step 0: validation input file
    raw_filesize = os.path.getsize(input_filename)
    if raw_filesize < 2000:
        return {'err_code': -2, 'err_msg': 'filesize < 2KB'}
    if os.path.splitext(input_filename)[1].lower() in sufset and raw_filesize < 4000:
        return {'err_code': -3, 'err_msg': 'file extension seems not a video'}

    if not os.path.isdir(tmp_dir):
        os.system("mkdir -p '%s' " % tmp_dir)
    else:
        os.system("rm '%s'/*.jpg" % tmp_dir)

    duration, input_fps = get_video_duration_fps(input_filename)

    if input_fps < 0.1 and duration > 7200.0:
        duration = input_fps*3600.0  # FIXME special for one bad case

    duration = duration if duration > 1.0 else 1.0

    picdir = tmp_dir  # os.path.join(tmp_dir, 'pics_cut')
    capture_num = 0
    out_list = []
    info = dict()
    err_code = 0
    err_msg = ''
    info['duration'] = duration

    if duration > 120:  # 300
        # step 1: transcode and save scene cut key frames yuv
        vf_scale = ''
        if width >= 16 or height >= 16:
            vf_scale = ' -vf scale=%d:%d' % (width, height)

        outfile = ''  # os.path.join(tmp_dir, 'out.mp4')  # -f mp4 -y /dev/null
        ffmpeg_video_params = "-preset superfast -timelimit 4000 -c:v libx264 -qmin 50 -x264opts picdir='%s' " % picdir

        if input_fps > 120:
            cmd_str = "%s -r 25 -v quiet -i '%s' -sn -an %s %s -pix_fmt yuv420p -metadata rotate=0 -r 25 -f null -" \
                      % (FFMPEG_PGC, input_filename, ffmpeg_video_params, vf_scale)
        else:
            cmd_str = "%s -v quiet -i '%s' -sn -an %s %s -pix_fmt yuv420p -metadata rotate=0 -f null -" \
                      % (FFMPEG_PGC, input_filename, ffmpeg_video_params, vf_scale)

        # print cmd_str
        (ret, _) = commands.getstatusoutput(cmd_str)
        if ret:
            err_msg += '[warning]fast scene cut get frames may failed!'

        # step 2: change yuv2jpg
        yuv_list = get_dir_pics_list(picdir, prefix='', suffix='.yuv')

        if width >= 16 and height >= 16:
            yuvsize = '%dx%d' % (width, height)  # input yuv size
        else:
            yuvsize = None

        for yuvfile in yuv_list:
            poc0 = yuv2jpg(yuvfile, vf=vf_filter, yuvsize=yuvsize)
            if poc0 is not None:
                out_list.append(poc0)

        capture_num = len(out_list)

    # print out_list
    out_list_final = out_list[:]

    if 1:
        # step 3: if scene cut frame less than 20, add another 20+ pictures
        add_pictures = 0
        if capture_num < max_fn and duration < 16:  # like hotsoon
            add_pictures = max_fn  # max_fn + 1
        # elif capture_num < 4 and duration < 16:  # like short duanzi aweme
        #    add_pictures = 4  # 4 + 2
        elif capture_num < 6 and duration < 20:  # like short duanzi
            add_pictures = 6  # 6 + 2
        elif capture_num < 10 and duration < 40:  # like short ugc
            add_pictures = 10  # 10 + 2
        elif capture_num < 20 or capture_num < duration/60:  # at least 60s get one picture, for some weixin ad
            if duration >= 1200:
                add_pictures = int(duration/40)
                if add_pictures > 50-capture_num:
                    add_pictures = 50-capture_num
            elif duration > 0:  # -1: N/A
                if capture_num > 0:
                    add_pictures = (duration + 5)/10  # if capture before, every 6s get 1 frame
                else:
                    add_pictures = (duration + 3)/6

                if add_pictures < 20-capture_num:
                    add_pictures = 20-capture_num
                elif add_pictures > 38-capture_num:
                    add_pictures = 38-capture_num
                else:
                    add_pictures = int(add_pictures)

        if add_pictures > 0:
            cfps = (add_pictures+0.1)/(duration+0.1)
            if capture_num > 0 and duration > 3.0:
                star_time = 2.0
            else:
                star_time = 0.0

            extra_pics = os.path.join(tmp_dir, 'extra_pics')
            os.system("mkdir -p '%s'" % extra_pics)
            out_list2 = ff_get_pictures_360p(input_filename, extra_pics, add_pictures, fps=cfps, vf=vf_filter, width=width, height=height, ss=star_time)
            extra_num = len(out_list2)
            # print out_list2

            if out_list2 and extra_num > 0:
                interval = duration * input_fps/extra_num  # frame_num_skip  4*fps  fixme it is just suit for constant fps
                interval = interval if interval >= 1.0 else 1.0  # special for fps < 1
                for pic in out_list2:
                    idx = os.path.basename(pic)
                    idx = idx[:idx.rfind('.')]
                    try:
                        idx = int(idx)
                        poc = int(star_time*input_fps + idx * interval)
                        if is_poc_inlist(out_list, poc, r=5):  # poc in out_list or (poc-1) in out_list or (poc+1) in out_list:
                            continue  # poc += 1 fixme if the picture has captured before
                        pic_new = os.path.join(picdir, '%d.jpg' % poc)
                        os.system("mv -f '%s' '%s'" % (pic, pic_new))
                        if poc not in out_list_final:
                            out_list_final.append(poc)
                    except:
                        pass
                err_msg += 'ext_num:%d ' % extra_num
            else:
                err_code = -2
                err_msg += '[warning]got no extra frames!'

            if os.path.isdir(extra_pics):
                os.system("rm -rf '%s'" % extra_pics)

        if len(out_list_final) > max_fn:  # not need for hotsoon
            first_jpg = ff_get_first_pic(input_filename, tmp_dir, width=width, height=height, suffix='0')

            if first_jpg:
                out_list_final.append(0)

        last_poc = int((duration + 0.5) * input_fps)
        if last_poc in out_list_final:
            last_poc += 10
        last_jpg = ff_get_last_pic(input_filename, tmp_dir, width=width, height=height, suffix='%s' % last_poc)
        if last_jpg:
            out_list_final.append(last_poc)

        out_list_final.sort()
        info['capture_frames'] = out_list_final[:]
        info['err_code'] = err_code
        info['err_msg'] = err_msg

        # as has no use_id we only do it for hotsoon <= 15s
        if duration < 20:
            video_ill = check_video_ill_probability(picdir, info['capture_frames'], width, height)
            info['ill'] = video_ill
        # print info
        return info


# only for hotsoon!
def check_video_ill_probability(picdir, poc_list, width=-1, height=-1):
    video_ill = 0
    total_pic = len(poc_list)
    if width > 16 and height > 16:
        thres = width * height / 22.0  # 640x360=230400
        thres = thres if thres < 8000 else 8000
    else:
        thres = 4000

    if total_pic > 0:
        ill_pic_num = 0
        ill_list = []
        # ill_path = os.path.join(picdir, 'ill')
        # os.system("mkdir -p '%s'" % ill_path)
        for poc in poc_list:
            pic = os.path.join(picdir, '%s.jpg' % poc)
            if os.path.isfile(pic):
                pic_size = os.path.getsize(pic)
                if pic_size < thres:
                    ill_pic_num += 1
                    ill_list.append(pic)
                    # os.system("mv '%s' '%s'" % (pic, ill_path))  # fixme just for debug

        if 1.0 * ill_pic_num / total_pic > 0.8:
            video_ill = 1

        # print 'ill probability [ill/total] = %d/%d,  illlist=%s' % (ill_pic_num, total_pic, str(ill_list))

    return video_ill


def main():
    param = dict()
    param['input'] = '/data/videos/x.mp4'
    param["tmp_dir"] = 'tmpx'

    # param["width"] = 640
    # param["height"] = 360
    outinfo = get_scenecut_frame(param)
    print outinfo


if __name__ == '__main__':
    main()

