import re

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# The broken classification block starts somewhere and ends with:
#       <div id="svm-status" class="mt-2 text-sm text-mute"></div>
#     </div>
# ...
#   </section> -->

pattern = re.compile(r'</select>\s*<input type="number" id="svm-c".*?</section>\s*-->', re.DOTALL)
content = pattern.sub('<div id="page-classification" class="page"></div>', content)

with open('index.html', 'w', encoding='utf-8') as f:
    f.write(content)
