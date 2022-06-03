def friendly_text(i):
    if i >= 1000000000000:
        return str(i / 1000000000000) + 'T'

    if i >= 1000000000:
        return str(i / 1000000000) + 'B'

    if i >= 1000000:
        return str(i / 1000000) + 'M'

    if i >= 1000:
        return str(i / 1000) + 'K'


print(friendly_text(2555))
print(friendly_text(241555))
print(friendly_text(241535555))
print(friendly_text(2415533347615))
print(friendly_text(2415537537355355515))

# will print:
# 2.555K
# 241.555K
# 241.535555M
# 2.415533347615T
# 2415537.5373553555T
