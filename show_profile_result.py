import pstats

p = pstats.Stats("profile.dmp")
p.strip_dirs().sort_stats('time').print_stats(10)
