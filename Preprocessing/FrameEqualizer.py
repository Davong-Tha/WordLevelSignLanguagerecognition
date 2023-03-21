def Sample_frame(frame_list, num_frameOut):
    output_frame = []
    total_frame = len(frame_list)
    i = 1
    while i != num_frameOut + 1:
        idx = int(i / num_frameOut * total_frame)
        frame = frame_list[idx]
        output_frame.append(frame)
        i += 1
    return output_frame
