N=int(input())
songs_new=input()
songs=songs_new.split()
repeated=[]
for i in range(N):
    if songs[i] in songs[i+1:]:
        repeated.append(songs[i])
print(len(repeated))