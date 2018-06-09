from __future__ import print_function
import midi
import numpy as np

file_path = "data/Marooned.mid"

pattern = midi.read_midifile(file_path)

for track in pattern:
    track_vec = []
    for event in track:
        if event.name == 'Note On':
            vec = [event.statusmsg, event.length, event.pitch, event.velocity, event.tick, event.data[0], event.data[1], event.channel]
            track_vec.append(vec)
    track_vec = np.array(track_vec)
    print(track_vec.shape)
