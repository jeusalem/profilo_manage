import re

pattern = re.compile(r'(.*)(\d{4}/\d{1,2}/\d{1,2})')

#match = pattern.match('hello 2016/8/12 111/23/4')
match = pattern.match('hello')
print(match)
if match:
    print(match.group(2))

s='12 34 st'
print(s.split())

