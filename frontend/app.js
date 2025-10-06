const apiBase = ""; // empty because nginx proxies /api to backend

const $ = (s)=>document.querySelector(s);
$("#btnHealth").onclick = async ()=>{
  try{
    const r = await fetch(`${apiBase}/api/health`);
    $("#health").textContent = r.ok ? "OK" : `HTTP ${r.status}`;
  }catch(e){ $("#health").textContent = "error"; }
};

$("#btnPredict").onclick = async ()=>{
  const text = $("#txt").value || "";
  const out = $("#out");
  out.textContent = "Loading...";
  try{
    const r = await fetch(`${apiBase}/api/predict`, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({text})
    });
    const data = await r.json();
    out.textContent = JSON.stringify(data, null, 2);
  }catch(e){
    out.textContent = String(e);
  }
};
