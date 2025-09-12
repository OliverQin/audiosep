#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, io, pathlib, sys
import numpy as np, pandas as pd
import mido

DEFAULT_TEMPO = 500000  # microseconds per beat (120 bpm)

def read_alignment_csv(path):
    df = pd.read_csv(path)
    # 兼容 TIME/Value/time/value 等
    lc = {c.lower(): c for c in df.columns}
    if "time" in lc and "value" in lc:
        t_live = df[lc["time"]].to_numpy(float)
        t_ref = df[lc["value"]].to_numpy(float)
    else:
        # 取前两列数值型
        num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
        assert len(num_cols) >= 2, "对齐CSV需要至少两列数值（TIME/Value）。"
        t_live = df[num_cols[0]].to_numpy(float)
        t_ref = df[num_cols[1]].to_numpy(float)
    # 排序去重
    order = np.argsort(t_ref)
    t_ref, t_live = t_ref[order], t_live[order]
    keep = np.concatenate(([True], np.diff(t_ref) > 1e-12))
    t_ref, t_live = t_ref[keep], t_live[keep]
    print(f"[AB] points={len(t_ref)}, A=[{t_ref[0]:.3f},{t_ref[-1]:.3f}]s -> B=[{t_live[0]:.3f},{t_live[-1]:.3f}]s")
    return t_ref, t_live

def interp_with_extrap(x, xp, fp):
    x = np.asarray(x, dtype=float)
    y = np.interp(x, xp, fp)
    if len(xp) >= 2:
        left = x < xp[0]
        right = x > xp[-1]
        if np.any(left):
            k = (fp[1]-fp[0])/(xp[1]-xp[0] + 1e-12)
            y[left] = fp[0] + k*(x[left]-xp[0])
        if np.any(right):
            k = (fp[-1]-fp[-2])/(xp[-1]-xp[-2] + 1e-12)
            y[right] = fp[-1] + k*(x[right]-xp[-1])
    return y

def load_midi_bytes(path):
    """支持常规SMF；若是 RIFF/RMID 则剥壳出 MThd 段后用 BytesIO 读。"""
    p = pathlib.Path(path)
    data = p.read_bytes()
    if data[:4] == b'RIFF' and b'MThd' in data:
        i = data.find(b'MThd')
        print("[MIDI] 检测到 RIFF/RMID，自动抽取 SMF...")
        return io.BytesIO(data[i:])
    return io.BytesIO(data)  # 也可直接传文件路径；这里统一用内存流

def merge_events(mid):
    """
    把所有轨的事件拉平成一个时间有序的列表：
    返回 [(abs_tick, track_idx, msg), ...]，按 abs_tick 升序稳定排序。
    """
    events = []
    for ti, track in enumerate(mid.tracks):
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            events.append((abs_tick, ti, msg))
    # 稳定排序：先按 abs_tick，再按 track_idx 保持一定确定性
    events.sort(key=lambda x: (x[0], x[1]))
    return events

def parse_track_names(mid):
    names = []
    for tr in mid.tracks:
        name = None
        for msg in tr:
            if msg.type == 'track_name':
                name = msg.name
                break
        names.append(name or "")
    return names

def notes_from_midi_with_mido(midi_path, track_sel=None, channel_sel=None,
                              name_substr=None, include_drums=False):
    """
    读取 MIDI，返回每个音的 (track_idx, channel, pitch, t_ref_on_sec, t_ref_off_sec)
    """
    mf = mido.MidiFile(file=load_midi_bytes(midi_path))
    tpb = mf.ticks_per_beat
    print(f"[MIDI] ticks_per_beat={tpb}, tracks={len(mf.tracks)}")

    track_names = parse_track_names(mf)
    def track_ok(ti):
        if track_sel is not None and ti != track_sel:
            return False
        if name_substr:
            return (name_substr.lower() in (track_names[ti] or "").lower())
        return True if track_sel is None else True

    # 全局遍历：合并事件 → 把 tick 积分成秒（处理 set_tempo）
    events = merge_events(mf)
    current_tempo = DEFAULT_TEMPO
    last_tick = 0
    time_sec = 0.0

    # 维护每个(channel, pitch)的栈，允许同音叠音
    stacks = {}  # key=(ch,pitch) -> list of (start_tick, start_sec, track_idx, include_flag)
    def key(ch, pitch): return (ch, pitch)

    # 程序变更（ProgramChange）跟踪（可用于后续扩展筛选），先简单记录最近一次
    program_by_channel = {ch: None for ch in range(16)}

    rows = []

    i = 0
    n = len(events)
    while i < n:
        tick_i = events[i][0]
        # 先把 last_tick -> tick_i 之间按当前 tempo 积分到秒
        if tick_i > last_tick:
            dticks = tick_i - last_tick
            time_sec += (dticks / tpb) * (current_tempo / 1e6)
            last_tick = tick_i

        # 处理同一 tick 上的所有消息
        j = i
        while j < n and events[j][0] == tick_i:
            _, ti, msg = events[j]
            if msg.type == 'set_tempo':
                # tempo 变更在该 tick 生效，影响后续时间推进
                current_tempo = msg.tempo
            elif msg.type == 'program_change':
                program_by_channel[msg.channel] = msg.program
            elif msg.type in ('note_on', 'note_off'):
                ch = msg.channel
                pitch = msg.note if hasattr(msg, 'note') else None
                if msg.type == 'note_on' and msg.velocity > 0:
                    # 过滤（轨/通道/鼓/名字）
                    if not include_drums and ch == 9:
                        inc = False
                    else:
                        if channel_sel is not None and ch != channel_sel:
                            inc = False
                        elif not track_ok(ti):
                            inc = False
                        else:
                            inc = True
                    stacks.setdefault(key(ch, pitch), []).append((tick_i, time_sec, ti, inc))
                else:
                    # note_off 或 note_on vel=0
                    k = key(ch, pitch)
                    if k in stacks and stacks[k]:
                        start_tick, start_sec, ti0, inc = stacks[k].pop()
                        if inc:
                            rows.append({
                                "track_idx": ti0,
                                "channel": ch,
                                "midi_note": int(pitch),
                                "t_ref_on": float(start_sec),
                                "t_ref_off": float(time_sec),
                            })
            # 其他消息忽略
            j += 1
        i = j

    # 清理未关音（以结尾时间作为 off）
    for (ch, pitch), lst in stacks.items():
        for start_tick, start_sec, ti0, inc in lst:
            if inc:
                rows.append({
                    "track_idx": ti0, "channel": ch, "midi_note": int(pitch),
                    "t_ref_on": float(start_sec), "t_ref_off": float(time_sec)
                })

    df = pd.DataFrame(rows).sort_values(["t_ref_on"]).reset_index(drop=True)
    if df.empty:
        raise RuntimeError("没有解析到任何音符（可能筛选条件过严，或 MIDI 不含 note）。")
    return df, track_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--midi", required=True, help="输入 MIDI 文件（支持 RMID 自动剥壳）")
    ap.add_argument("--align_csv", required=True, help="MATCH A→B 时间线 CSV（TIME,VALUE）")
    ap.add_argument("--out", default="notes_live.csv")
    ap.add_argument("--out_full", default=None)
    # 选择 & 过滤
    ap.add_argument("--track", type=int, default=None, help="仅取第 N 个轨（0-based）")
    ap.add_argument("--channel", type=int, default=None, help="仅取某个通道（0..15）")
    ap.add_argument("--name_substr", type=str, default=None, help="按轨名包含关键字过滤")
    ap.add_argument("--include_drums", action="store_true", help="包含鼓（channel 10）")
    # 形态处理
    ap.add_argument("--min_dur", type=float, default=0.0, help="把短于该值(秒)的音扩到最小时长")
    ap.add_argument("--monophonic", action="store_true", help="单声部化：同一开始时刻仅保留最高音")
    args = ap.parse_args()

    # A→B 映射
    t_ref, t_live = read_alignment_csv(args.align_csv)
    map_fn = lambda x: interp_with_extrap(x, t_ref, t_live)

    # 解析 MIDI → 参考时间
    df, track_names = notes_from_midi_with_mido(
        args.midi,
        track_sel=args.track,
        channel_sel=args.channel,
        name_substr=args.name_substr,
        include_drums=args.include_drums
    )

    from IPython import embed; embed()  # DEBUG

    # 映射到 live 时间轴
    df["time_start"] = map_fn(df["t_ref_on"].to_numpy())
    df["time_end"]   = map_fn(df["t_ref_off"].to_numpy())

    # 最小时长
    if args.min_dur and args.min_dur > 0:
        short = (df["time_end"] - df["time_start"]) < args.min_dur
        df.loc[short, "time_end"] = df.loc[short, "time_start"] + args.min_dur

    # 单声部化（把相同起点的重叠合并为一个，取最高音）
    if args.monophonic:
        bucket = (df["time_start"]/0.005).round().astype(int)  # 5ms 桶
        df["_b"] = bucket
        df.sort_values(["_b","midi_note"], ascending=[True, False], inplace=True)
        df = df.groupby("_b", as_index=False).first().drop(columns=["_b"]).sort_values("time_start")

    # 导出
    simple = df[["midi_note","time_start","time_end"]].copy()
    simple.columns = ["note","time_start","time_end"]
    simple.to_csv(args.out, index=False)

    if args.out_full:
        # 附带来源轨/通道与参考时间，便于排查
        out_full_cols = ["track_idx","channel","midi_note","t_ref_on","t_ref_off","time_start","time_end"]
        if "track_idx" in df.columns:
            df["track_name"] = df["track_idx"].map(lambda i: track_names[i] if 0 <= i < len(track_names) else "")
            out_full_cols = ["track_idx","track_name","channel","midi_note","t_ref_on","t_ref_off","time_start","time_end"]
        df[out_full_cols].to_csv(args.out_full, index=False)

    dur = (simple["time_end"] - simple["time_start"]).to_numpy()
    print(f"[OUT] notes={len(simple)}, t=[{simple['time_start'].min():.3f},{simple['time_end'].max():.3f}]s, "
          f"dur(min/med/max)={dur.min():.3f}/{np.median(dur):.3f}/{dur.max():.3f}s")
    if args.track is not None:
        try:
            print(f"[INFO] 使用轨 {args.track}: “{track_names[args.track]}”")
        except Exception:
            pass

if __name__ == "__main__":
    main()
