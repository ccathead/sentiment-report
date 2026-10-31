// Execute the real template's UI logic with a minimal DOM; no user data is loaded.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const template = fs.readFileSync(path.join(__dirname, '../templates/review_template.html'), 'utf8');
const script = template.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function boot(rows, records = {}) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, {value:'', textContent:'', style:{}, children:[],
      addEventListener(){}, removeAttribute(){}, classList:{toggle(){}},
      appendChild(option){this.children.push(option);if(!this.value)this.value=option.value;}});
    return elements.get(id);
  }
  element('dataset').textContent = JSON.stringify({annotations:rows,annotations_sha256:'same-version'});
  const store = new Map([['mvsa-review-same-version', JSON.stringify({reviewer:'reviewer',records})]]);
  const document = {getElementById:element, createElement:()=>({}), querySelector:element,
    querySelectorAll:()=>[], addEventListener(){}};
  vm.runInNewContext(script, {document, window:{addEventListener(){}},
    localStorage:{getItem:key=>store.get(key),setItem:(key,value)=>store.set(key,value)},
    alert(){}, confirm:()=>true});
  return {element,store};
}
const rows = Array.from({length:360},(_,i)=>({id:`sample-${i}`,split:'candidate',text:'post',image:'image.jpg'}));
const ui = boot(rows);
assert.equal(ui.element('scope').children[0].textContent,'全部 360 条');
assert.equal(ui.element('progress').textContent,'已复核 0/360；待复核 360');
assert.match(ui.element('identity').textContent,/训练／验证候选（待划分）/);
assert.equal(ui.element('exportComplete').disabled,true);
assert.doesNotMatch(ui.element('scope').children.map(o=>o.textContent).join(),/开发|挑战|72/);
assert.match(ui.element('proposal').textContent,/未生成 AI/);
const draft = { 'sample-0':{label:'positive',reviewer:'reviewer',reviewed_at:'2026-10-31'}};
const refreshed = boot(rows,draft);
assert.match(refreshed.element('progress').textContent,/已复核 1\/360/);
assert.equal(refreshed.element('reviewer').value,'reviewer');
refreshed.element('scope').value='reviewed';refreshed.element('scope').onchange();
assert.match(refreshed.element('identity').textContent,/sample-0/);
const complete = Object.fromEntries(rows.map(r=>[r.id,{label:'neutral'}]));
assert.equal(boot(rows,complete).element('exportComplete').disabled,false);
const legacy = boot(['dev','test','challenge'].map((split,i)=>({id:`old-${i}`,split})));
assert.equal(legacy.element('scope').children.find(o=>o.value==='eval').textContent,'评测集 3 条');
console.log('Review page checks passed: dynamic counts, roles, filters, completeness, preserved drafts.');
