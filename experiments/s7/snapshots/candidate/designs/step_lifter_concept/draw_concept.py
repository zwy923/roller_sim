"""SF-01 R0: vector concept drawings, not a fabrication release or simulation.

All machine coordinates are millimetres. z=0 is the existing receiving belt.
The draft dimensions below are design proposals, not measured equipment data.
"""
from pathlib import Path
import math
import json
import html
import zipfile
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'pdf' / 'step_lifter_concept'
OUT.mkdir(parents=True, exist_ok=True)
FONT_PATH = Path('C:/Windows/Fonts/simhei.ttf')
pdfmetrics.registerFont(TTFont('CJK', str(FONT_PATH)))
MM = 72 / 25.4
INK, MUTED = '#233344', '#617383'
BLUE, LIGHT = '#167baf', '#e6f2f9'
ORANGE, RED = '#bb7018', '#ac3e32'
GRAY, PALE, GREEN = '#a9b3bd', '#f2f5f7', '#25826d'
P = dict(feed_length=1800, feed_width=1200, pickup_opening=920,
         carrier_width=900, pad_width=500, pad_depth=300,
         alpha_deg=30, shoulder_deg=15, feed_deg=15, stroke=650,
         toe_low_z=-520, lip_z=100, outlet_gap=15, inlet_gap=10,
         chute_length=200, outlet_width=920, belt_width=1200,
         frame_width=1500, frame_top_z=2050, frame_base_z=-1250,
         apron_height=760, crosshead_low_z=480, rod_base_z=1230)
P['heel_low_z'] = P['toe_low_z'] + P['pad_depth'] * math.tan(math.radians(P['alpha_deg']))
P['toe_high_z'] = P['toe_low_z'] + P['stroke']
P['heel_high_z'] = P['heel_low_z'] + P['stroke']
P['shoulder_front_low_z'] = P['heel_low_z'] + P['pad_depth'] * math.tan(math.radians(P['shoulder_deg']))
P['lip_x'] = P['feed_length'] + P['pad_depth'] + P['outlet_gap']
P['belt_x'] = P['lip_x'] + P['chute_length']
P['feed_back_z'] = P['heel_low_z'] + (P['feed_length']-P['inlet_gap']) * math.tan(math.radians(P['feed_deg']))

class Sheet:
    def __init__(self, pdf, number, title):
        self.c, self.n = pdf, number
        self.svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">',
                    '<rect width="420" height="297" fill="white"/>']
        self.rect(8, 8, 404, 281, stroke=INK, lw=.35)
        self.text(15, 279, title, 6.1)
        self.text(15, 271.5, 'SF-01  /  单级窄托台抬料器  /  R0 结构方案', 3.5, MUTED)
        self.text(405, 279, '2026-09-22', 3.4, MUTED, 'right')
        self.text(405, 271.5, '方案尺寸待验证 · 非加工图', 3.7, RED, 'right')
        self.line((8, 266), (412, 266), INK, .3)
        self.rect(8, 8, 404, 26, stroke=INK, lw=.3)
        for x in [204, 299, 358]:
            self.line((x, 8), (x, 34), INK, .25)
        self.text(13, 24, '五块随机煤矸 · 抬料提取候选结构', 4.0)
        self.text(13, 15, '原主线保持原位；新增模块在现有主带上游接入。', 3.0, MUTED)
        self.text(209, 25, '单位：mm    比例：见各视图', 3.2)
        self.text(209, 17, '* 为候选值；不得据此采购或加工', 3.1, RED)
        self.text(304, 25, '主线接口：带面 Z=0', 3.1)
        self.text(304, 17, '图示安装面：Z=-1250*', 3.1)
        self.text(363, 25, f'SF-01-0{number}', 4.0)
        self.text(363, 16, f'R0     {number} / 2', 3.2, MUTED)

    @staticmethod
    def y(y): return 297-y

    def line(self, a, b, color=INK, lw=.35, dash=None):
        c = self.c
        c.setStrokeColor(color); c.setLineWidth(lw*MM)
        c.setDash([x*MM for x in dash] if dash else [])
        c.line(a[0]*MM, a[1]*MM, b[0]*MM, b[1]*MM)
        ds = f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else ''
        self.svg.append(f'<line x1="{a[0]:.4f}" y1="{self.y(a[1]):.4f}" x2="{b[0]:.4f}" y2="{self.y(b[1]):.4f}" stroke="{color}" stroke-width="{lw}"{ds}/>')

    def poly(self, pts, color=INK, fill=None, lw=.35, close=True, dash=None):
        c = self.c; c.setLineWidth(lw*MM); c.setStrokeColor(color)
        c.setDash([x*MM for x in dash] if dash else [])
        if fill: c.setFillColor(fill)
        p = c.beginPath(); p.moveTo(pts[0][0]*MM, pts[0][1]*MM)
        for x,y in pts[1:]: p.lineTo(x*MM,y*MM)
        if close:p.close()
        c.drawPath(p, stroke=1, fill=int(bool(fill)))
        ds = f' stroke-dasharray="{",".join(map(str,dash))}"' if dash else ''
        tag = 'polygon' if close else 'polyline'
        self.svg.append(f'<{tag} points="'+ ' '.join(f'{x:.4f},{self.y(y):.4f}' for x,y in pts)+f'" fill="{fill or "none"}" stroke="{color}" stroke-width="{lw}"{ds}/>')

    def rect(self,x,y,w,h,fill=None,stroke=INK,lw=.3,dash=None):
        self.poly([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],stroke,fill,lw,True,dash)

    def circle(self,x,y,r,color=INK,fill=None,lw=.3):
        self.c.setStrokeColor(color);self.c.setLineWidth(lw*MM);self.c.setDash([])
        if fill:self.c.setFillColor(fill)
        self.c.circle(x*MM,y*MM,r*MM,stroke=1,fill=int(bool(fill)))
        self.svg.append(f'<circle cx="{x}" cy="{self.y(y)}" r="{r}" stroke="{color}" stroke-width="{lw}" fill="{fill or "none"}"/>')

    def text(self,x,y,text,size=3.3,color=INK,anchor='left',rotate=0):
        c=self.c;c.saveState();c.translate(x*MM,y*MM);c.rotate(rotate)
        c.setFillColor(color);c.setFont('CJK',size*MM)
        {'left':c.drawString,'center':c.drawCentredString,'right':c.drawRightString}[anchor](0,0,text)
        c.restoreState()
        anc={'left':'start','center':'middle','right':'end'}[anchor]
        self.svg.append(f'<text x="{x}" y="{self.y(y)}" font-family="SimHei, Microsoft YaHei, sans-serif" font-size="{size}" fill="{color}" text-anchor="{anc}" transform="rotate({-rotate} {x} {self.y(y)})">{html.escape(text)}</text>')

    def arrow(self,a,b,color=BLUE,lw=.55,double=False):
        self.line(a,b,color,lw)
        def tip(p,q):
            v=(p[0]-q[0],p[1]-q[1]);d=math.hypot(*v);ux,uy=v[0]/d,v[1]/d
            self.poly([p,(p[0]-2.1*ux+0.8*uy,p[1]-2.1*uy-0.8*ux),(p[0]-2.1*ux-0.8*uy,p[1]-2.1*uy+0.8*ux)],color,color,.2)
        tip(b,a)
        if double:tip(a,b)

    def dimh(self,x1,x2,y,from_y,label):
        for x in [x1,x2]:self.line((x,from_y),(x,y+2 if y>from_y else y-2),MUTED,.18)
        self.arrow((x1,y),(x2,y),MUTED,.2,True)
        tw=pdfmetrics.stringWidth(label,'CJK',3.0*MM)/MM
        self.rect((x1+x2)/2-tw/2-1,y+.6,tw+2,4.3,fill='#ffffff',stroke='#ffffff',lw=0)
        self.text((x1+x2)/2,y+1.3,label,3.0,MUTED,'center')

    def dimv(self,y1,y2,x,from_x,label):
        for y in [y1,y2]:self.line((from_x,y),(x+2,y),MUTED,.18)
        self.arrow((x,y1),(x,y2),MUTED,.2,True)
        self.text(x-1.6,(y1+y2)/2,label,3.0,MUTED,'center',90)

    def label(self,where,knee,end,text,color=INK):
        self.poly([where,knee,end],color,None,.25,False)
        self.circle(*where,.6,color,color)
        self.text(end[0],end[1]+1.6,text,3.2,color)

    def note(self,x,y,lines,size=3.1,leading=5,color=INK):
        for i,line in enumerate(lines):self.text(x,y-i*leading,line,size,color)

    def finish(self):
        self.svg.append('</svg>')
        (OUT/f'SF-01-R0-sheet-{self.n}.svg').write_text('\n'.join(self.svg),encoding='utf-8')
        self.c.showPage()

def side_shapes(stroke=0):
    x=P['feed_length'];d=P['pad_depth'];h=P['heel_low_z']+stroke;n=P['toe_low_z']+stroke
    return {
      'pan': [(0,P['feed_back_z']),(x-10,P['heel_low_z']),(x-10,P['heel_low_z']-20),(0,P['feed_back_z']-20)],
      'pad': [(x,h),(x+d,n),(x+d,n-20),(x,h-20)],
      'wall': [(P['lip_x'],P['lip_z']),(P['lip_x']+20,P['lip_z']),(P['lip_x']+20,-690),(P['lip_x'],-690)],
      'chute': [(P['lip_x'],100),(P['belt_x'],0),(P['belt_x'], -15),(P['lip_x'],85)],
      'belt': [(P['belt_x'],0),(2700,0),(2700,-65),(P['belt_x'],-65)],
      'carriage': [(1795,-665+stroke),(2110,-665+stroke),(2110,-625+stroke),(1795,-625+stroke)],
      'apron': [(1800,h),(1820,h-20),(1820,h-760),(1800,h-760)],
    }

def world_poly(s,T,pts,color=INK,fill=None,lw=.35,dash=None):
    s.poly([T(*p) for p in pts],color,fill,lw,True,dash)

def side(s,T,stroke=0,mast=True,ghost=False,frame=True):
    sh=side_shapes(stroke)
    for k in ['pan','wall','chute','belt']:
        world_poly(s,T,sh[k],GRAY if k=='belt' else INK,PALE,.3)
    for k in ['apron','pad','carriage']:world_poly(s,T,sh[k],BLUE,LIGHT,.45)
    # Side shoulder slopes back toward the pan, unlike the central discharge face.
    s.line(T(1800,P['heel_low_z']+stroke),T(2100,P['shoulder_front_low_z']+stroke),ORANGE,.35,[1.2,.8])
    if frame:
        for x in [160,1730,2140]:
            ztop=(P['feed_back_z']-20-160*math.tan(math.radians(15))) if x==160 else -570
            world_poly(s,T,[(x,-1250),(x+55,-1250),(x+55,ztop),(x,ztop)],GRAY,None,.25)
        s.line(T(-100,-1250),T(2400,-1250),MUTED,.2,[3,1,1,1])
    if mast:
        world_poly(s,T,[(1700,-1250),(1760,-1250),(1760,2050),(1700,2050)],GRAY,None,.25)
        world_poly(s,T,[(2140,-1250),(2200,-1250),(2200,2050),(2140,2050)],GRAY,None,.25)
        world_poly(s,T,[(1700,1970),(2200,1970),(2200,2050),(1700,2050)],INK,PALE,.3)
        world_poly(s,T,[(1910,1230),(1990,1230),(1990,1950),(1910,1950)],INK,PALE,.4)
        s.line(T(1950,1950),T(1950,1970),INK,.5)
        s.line(T(1950,1230),T(1950,480+stroke),BLUE,.75)
        world_poly(s,T,[(1740,460+stroke),(2170,460+stroke),(2170,500+stroke),(1740,500+stroke)],BLUE,LIGHT,.35)
        for xx in [1800,2070]:
            s.line(T(xx,460+stroke),T(xx,-625+stroke),BLUE,.45,[2,1])
    if ghost:
        gh=side_shapes(650)
        world_poly(s,T,gh['apron'],BLUE,None,.25,[2,1])
        world_poly(s,T,gh['pad'],BLUE,None,.3,[2,1])
        world_poly(s,T,gh['carriage'],BLUE,None,.25,[2,1])
        if mast:s.line(T(1740,480+650),T(2170,480+650),BLUE,.3,[2,1])
    s.line(T(-100,0),T(2760,0),MUTED,.2,[4,1,1,1])

def rock(s,T,cx,cz,w=340,h=310,fill='#e7dfd1',label=None, surface_z=None, slope=None):
    pts=[(-.49,-.18),(-.34,-.46),(.18,-.49),(.48,-.24),(.43,.25),(.15,.5),(-.3,.44),(-.5,.13)]
    if surface_z is not None:
        cz=surface_z-min(h*py+slope*w*px for px,py in pts)
    world_poly(s,T,[(cx+w*x,cz+h*y) for x,y in pts],ORANGE,fill,.35)
    if label:s.text(*T(cx,cz),label,3.3,ORANGE,'center')

def main_sheet(pdf):
    s=Sheet(pdf,1,'总装布置与接口尺寸')
    # Side view: exact drawing scale 1:20 at 100% A3 print size.
    T=lambda x,z:(30+x/20,121+z/20)
    s.text(16,256,'A-A 纵向剖视  /  1:20',4.1)
    side(s,T,0,True,True)
    rock(s,T,1930,0,340,310,label='煤',surface_z=P['heel_low_z']-130*math.tan(math.radians(30)),slope=math.tan(math.radians(30)))
    s.dimh(*[T(x,0)[0] for x in [0,1800]],51,T(0,-1250)[1],'1800* 料盘至托台后缘')
    s.dimh(T(1800,0)[0],T(2100,0)[0],78,T(0,-650)[1],'D=300*')
    s.dimv(T(0,-520)[1],T(0,130)[1],199,T(2100,0)[0],'S=650* 升程')
    s.dimv(T(0,-1250)[1],T(0,2050)[1],19,T(0,0)[0],'3300* 安装包络')
    s.label(T(1950,1620),(142,238),(122,238),'01 上置提升执行器')
    s.label(T(1730,900),(127,175),(80,175),'02 门架与外置双侧导向')
    s.label(T(700,P['feed_back_z']-700*math.tan(math.radians(15))),(65,137),(34,137),'03 浅料盘：底坡15°*')
    s.label(T(1950,-433),(107,87),(59,87),'04 可换中央托面：30°*',BLUE)
    s.label(T(1950,216),(136,154),(117,154),'高位轮廓（虚线）',BLUE)
    s.label(T(2130,-300),(178,103),(167,103),'05 固定挡料面')
    s.label(T(1810,-860),(153,65),(147,65),'07 随动挡料裙板',BLUE)
    s.text(154,132,'06 接料斜板',3.0)
    s.text(164,115,'接现有主带 →',3.0,MUTED)
    s.text(33,123,'Z=0 既有带面',3.1,MUTED)
    s.note(30,236,['单个执行器只做升降。','裙板随托台运动，升起时封住下方开口。'],3.1,5,BLUE)
    s.text(33,43,'门架立柱、导向及吊杆位于料道两侧；剖视采用投影示意。',2.8,MUTED)
    # Plan view.
    Q=lambda x,y:(237+x/15,210+y/15)
    s.text(237,256,'俯视布置  /  1:15',4.1)
    pan=[(0,-600),(1200,-600),(1800,-460),(1800,460),(1200,600),(0,600)]
    world_poly(s,Q,pan,INK,PALE)
    world_poly(s,Q,[(1800,-450),(2100,-450),(2100,450),(1800,450)],BLUE,LIGHT)
    world_poly(s,Q,[(1800,-250),(2100,-250),(2100,250),(1800,250)],BLUE,'#c9e6f5',.5)
    for a,b in [(-450,-250),(250,450)]:
        s.arrow(Q(2040,(a+b)/2),Q(1830,(a+b)/2),ORANGE,.4)
    world_poly(s,Q,[(2115,-460),(2315,-460),(2315,460),(2115,460)],INK,PALE)
    world_poly(s,Q,[(2315,-600),(2480,-600),(2480,600),(2315,600)],GRAY,None,.25)
    s.line(Q(-70,0),Q(2530,0),MUTED,.2,[4,1,1,1])
    for xx,yy,ww,hh in [(400,280,360,340),(440,-260,430,360),(1000,280,420,350),(1060,-260,360,360),(1570,10,390,350)]:
        rock(s,Q,xx,yy,ww,hh)
    s.dimh(Q(1800,0)[0],Q(2100,0)[0],169,Q(0,-460)[1],'D=300*')
    s.dimv(Q(0,-600)[1],Q(0,600)[1],233,Q(0,0)[0],'1200 料盘宽*')
    s.text(247,214,'五块仅作装料示意',2.8,MUTED)
    s.text(357,245,'回落斜肩',2.9,ORANGE)
    s.text(346,165,'中央 W=500*；外宽900*',3.0,BLUE)
    # Front elevation, x is transverse y; z is vertical.
    F=lambda y,z:(272+y/30,85+z/30)
    s.text(238,158,'出料端正视  /  1:30',3.9)
    for y in [-750,670]:world_poly(s,F,[(y,-1250),(y+80,-1250),(y+80,2050),(y,2050)],INK,PALE,.3)
    world_poly(s,F,[(-750,1970),(750,1970),(750,2050),(-750,2050)],INK,PALE)
    world_poly(s,F,[(-40,1230),(40,1230),(40,1950),(-40,1950)],INK,PALE)
    s.line(F(0,1950),F(0,1970),INK,.5)
    s.line(F(0,1230),F(0,480),BLUE,.8)
    world_poly(s,F,[(-670,460),(670,460),(670,500),(-670,500)],BLUE,LIGHT)
    for y in [-650,650]:s.line(F(y,460),F(y,-640),BLUE,.45)
    world_poly(s,F,[(-670,-665),(670,-665),(670,-625),(-670,-625)],BLUE,LIGHT)
    world_poly(s,F,[(-450,-540),(450,-540),(450,-520),(-450,-520)],BLUE,LIGHT)
    world_poly(s,F,[(-450,P['heel_low_z']-760),(450,P['heel_low_z']-760),(450,P['heel_low_z']),(-450,P['heel_low_z'])],BLUE,None,.25,[1.3,.7])
    for z in [100,0]:s.line(F(-460,z),F(460,z),GRAY,.3)
    s.line(F(0,-1290),F(0,2070),MUTED,.2,[3,1,1,1])
    s.dimh(F(-750,0)[0],F(750,0)[0],39,F(0,-1250)[1],'1500* 门架外宽')
    # Parameter and status box.
    s.rect(320,40,84,112,fill=PALE,stroke=GRAY,lw=.2)
    s.text(324,144,'R0 候选尺寸 / 边界',3.9)
    rows=[('中央承托宽 W','400 / 500 / 600*'),('承托进深 D','300*'),('升程 S','650*'),('中央卸料坡 α','30°*'),('斜肩回料坡 β','15°*，朝上游'),('高位鼻端 Z','+130*'),('固定出料唇 Z','+100*'),('斜板落差 / 长','100 / 200*')]
    for i,(k,v) in enumerate(rows):
        yy=135-i*7
        s.text(324,yy,k,3.1);s.text(400,yy,v,3.0,BLUE,'right')
    s.note(324,72,['外形约2500×1500×3300*。','需带面下方约1250*安装空间；','高度不符时须重做接口方案。','板厚、型材、缸径、焊缝均未定。','本图不证明“每次只有一块”。'],2.9,5,RED)
    s.finish()

def detail_sheet(pdf):
    s=Sheet(pdf,2,'动作原理、承托面与设计边界')
    for idx,(stroke,title,notes) in enumerate([
        (0,'① 低位接料',['托台后缘与料盘接近齐平。','五块料在浅盘内；图中只画局部。']),
        (350,'② 抬离料群',['随动裙板封住托台下方开口。','斜肩向上游回料；不保证排除双块。']),
        (650,'③ 越唇卸料',['中央托面越过固定出料唇。','利用斜面卸料，下一次等待下游清空。'])]):
        left=15+idx*132
        s.rect(left,142,126,117,stroke=GRAY,lw=.2)
        s.text(left+5,250,title,4.3,BLUE)
        T=lambda x,z,l=left:(l+6+(x-1100)/15,193+z/15)
        # Crop the feed pan at x=1150 for this local action view.
        sh=side_shapes(stroke)
        z1150=P['heel_low_z']+(1790-1150)*math.tan(math.radians(15))
        world_poly(s,T,[(1150,z1150),(1790,P['heel_low_z']),(1790,P['heel_low_z']-20),(1150,z1150-20)],INK,PALE)
        for k in ['wall','chute','belt']:world_poly(s,T,sh[k],GRAY if k=='belt' else INK,PALE)
        # Only show the local portion of the long moving apron in these cropped views.
        ap=sh['apron']
        clipped=[(xx,max(zz,-660)) for xx,zz in ap]
        world_poly(s,T,clipped,BLUE,LIGHT,.35)
        world_poly(s,T,sh['pad'],BLUE,LIGHT,.5)
        s.line(T(1800,P['heel_low_z']+stroke),T(2100,P['shoulder_front_low_z']+stroke),ORANGE,.35,[1.5,1])
        if idx<2:
            rock(s,T,1930,0,340,310,label='A',surface_z=P['heel_low_z']+stroke-130*math.tan(math.radians(30)),slope=math.tan(math.radians(30)))
        else:
            rock(s,T,2240,0,320,300,label='A',surface_z=37.5,slope=.5)
            s.arrow(T(2110,340),T(2380,190),BLUE,.65)
        rock(s,T,1390,0,350,300,fill='#f2eee7',label='余料',surface_z=P['heel_low_z']+(1790-1390)*math.tan(math.radians(15)),slope=math.tan(math.radians(15)))
        if idx==1:s.arrow(T(2170,-350),T(2170,130),BLUE,.65)
        s.line(T(1100,0),T(2820,0),MUTED,.2,[3,1,1,1])
        s.note(left+5,153,notes,3.0,5)
    # Detail A: plan of replaceable pad and return shoulders.
    s.text(15,132,'B  承托面展开 / 俯视  1:7.5',4.0)
    T=lambda y,x:(26+(y+450)/7.5,72+(x-1800)/7.5)
    for ya,yb,fill,col in [(-450,-250,'#fbefdc',ORANGE),(-250,250,LIGHT,BLUE),(250,450,'#fbefdc',ORANGE)]:
        world_poly(s,T,[(ya,1800),(yb,1800),(yb,2100),(ya,2100)],col,fill,.45)
    s.arrow(T(0,1840),T(0,2070),BLUE,.7)
    for yy in [-350,350]:s.arrow(T(yy,2050),T(yy,1850),ORANGE,.65)
    s.text(86,89,'中央托面',3.2,BLUE,'center')
    s.text(38,101,'回料',2.8,ORANGE,'center');s.text(133,101,'回料',2.8,ORANGE,'center')
    s.dimh(T(-450,0)[0],T(450,0)[0],62,72,'900* 托台总宽')
    s.dimh(T(-250,0)[0],T(250,0)[0],121,112,'W=500* 可换')
    s.dimv(72,112,154,146,'D=300*')
    s.note(16,53,['中央向下游倾30°*；两侧向上游倾15°*。','W换400/500/600时，同时更换配套斜肩。','改D或坡角时，必须同步重算高位及接口。'],2.9,5)
    # Local gap geometry, intentionally no fabrication tolerance approval.
    s.text(173,132,'C  出料唇局部  /  示意放大',4.0)
    U=lambda x,z:(178+(x-2010)/3,76+(z-20)/3)
    world_poly(s,U,[(2010,182),(2100,130),(2100,110),(2010,162)],BLUE,LIGHT,.5)
    world_poly(s,U,[(2115,100),(2145,85),(2145,-5),(2115,-5)],INK,PALE,.5)
    s.arrow(U(2050,80),U(2050,160),BLUE,.6)
    s.label(U(2100,130),(219,119),(210,119),'高位鼻端 +130*',BLUE)
    s.label(U(2115,100),(240,104),(229,104),'固定唇 +100*')
    for xx,zz in [(2100,130),(2115,100)]:s.line(U(xx,zz),(U(xx,zz)[0],64),MUTED,.2)
    s.arrow((U(2100,0)[0],66),(U(2115,0)[0],66),MUTED,.2,True)
    s.line((194,66),(U(2100,0)[0],66),MUTED,.2)
    s.text(178,65,'g=15*',3.0,MUTED)
    s.note(171,57,['g仅为几何占位，不是防夹合格间隙。','缝口需可更换柔性刮片/遮挡，避免碎屑进入。','下行前确认空台；接料后缘同样存在闭合夹点。'],2.85,5,RED)
    # Review/parts table.
    s.rect(282,40,124,96,fill=PALE,stroke=GRAY,lw=.2)
    s.text(287,127,'零部件与验证边界',3.8)
    s.note(287,119,[
        '01 提升执行器 ×1：缸径、推拉力、保持方式待定。',
        '02 门架 + 双侧导向：承偏载，导向置于料道外。',
        '03 浅料盘：可换低摩擦衬板；底坡候选15°。',
        '04 托台 + 吊架：中央承托面和两侧斜肩可换。',
        '05 固定挡料面：低位挡住料；与托台留运动间隙。',
        '06 斜板：长200、落差100；07 随动裙板高760。',
        '必要检测：上下限位、过载、空台/出料确认。',
        '普通对射只判断有无料，不能证明只有一块。',
        '首测：单取率、双取率、空取率、夹料及破碎。',
        '风险：两块同托、宽料盘滞留/起拱、回落碰撞。',
        '五块总重、最重单块、摩擦及安装标高均需实测。',
        '图示煤块不是仿真结果；未完成承载与全行程校核。',
    ],2.85,5.5)
    s.text(16,37,'原理参考：Performance Feeders / Step Feeders（先提一排，再分选）；尺寸为本方案构想，非厂商参数。',2.55,MUTED)
    s.c.linkURL('https://performancefeeders.com/products/step-feeders',(15*MM,35*MM,278*MM,40*MM),relative=0)
    s.finish()

def dxf_export():
    """Minimal R2000 ASCII DXF. Three views use 1:1 millimetres, separated in modelspace."""
    records=[]
    def add(code,value):records.extend([str(code),str(value)])
    for code,val in [(0,'SECTION'),(2,'HEADER'),(9,'$ACADVER'),(1,'AC1015'),(9,'$INSUNITS'),(70,4),(0,'ENDSEC'),(0,'SECTION'),(2,'TABLES'),(0,'TABLE'),(2,'LAYER'),(70,5)]:add(code,val)
    for name,col in [('FIXED',7),('MOVING_LOW',5),('MOVING_HIGH',4),('DATUM',8),('NOTES',3)]:
        for code,val in [(0,'LAYER'),(2,name),(70,0),(62,col),(6,'CONTINUOUS')]:add(code,val)
    for code,val in [(0,'ENDTAB'),(0,'ENDSEC'),(0,'SECTION'),(2,'ENTITIES')]:add(code,val)
    def line(a,b,layer='FIXED'):
        for code,val in [(0,'LINE'),(8,layer),(10,a[0]),(20,a[1]),(30,0),(11,b[0]),(21,b[1]),(31,0)]:add(code,val)
    def poly(pts,layer='FIXED'):
        for a,b in zip(pts,pts[1:]+pts[:1]):line(a,b,layer)
    def text(x,y,t,h=45):
        for code,val in [(0,'TEXT'),(8,'NOTES'),(10,x),(20,y),(30,0),(40,h),(1,t),(50,0)]:add(code,val)
    sh=side_shapes(0)
    for k in ['pan','wall','chute','belt']:poly(sh[k])
    for st,lay in [(0,'MOVING_LOW'),(650,'MOVING_HIGH')]:
        shapes=side_shapes(st)
        for k in ['apron','pad','carriage']:poly(shapes[k],lay)
        line((1800,P['heel_low_z']+st),(2100,P['shoulder_front_low_z']+st),lay)
        poly([(1740,460+st),(2170,460+st),(2170,500+st),(1740,500+st)],lay)
    for x in [1700,2140]:poly([(x,-1250),(x+60,-1250),(x+60,2050),(x,2050)])
    poly([(1700,1970),(2200,1970),(2200,2050),(1700,2050)])
    poly([(1910,1230),(1990,1230),(1990,1950),(1910,1950)])
    line((1950,1950),(1950,1970))
    line((-150,0),(2800,0),'DATUM')
    text(0,2200,'SF-01 R0 / SIDE VIEW / MM 1:1 / CONCEPT ONLY',60)
    text(0,2080,'Stroke 650; pad D300; Z low nose -520 / high +130; fixed lip +100',40)
    text(0,-1420,'NO LOAD RATING / NO FABRICATION TOLERANCES / NOT SINGLE-PIECE VALIDATED',38)
    # Plan coordinates map flow x to DXF x, transverse y to DXF y-3000.
    def plan(pts,layer='FIXED'):poly([(x,y-3000) for x,y in pts],layer)
    plan([(0,-600),(1200,-600),(1800,-460),(1800,460),(1200,600),(0,600)])
    plan([(1800,-450),(2100,-450),(2100,450),(1800,450)],'MOVING_LOW')
    plan([(1800,-250),(2100,-250),(2100,250),(1800,250)],'MOVING_HIGH')
    plan([(2115,-460),(2315,-460),(2315,460),(2115,460)])
    plan([(2315,-600),(2700,-600),(2700,600),(2315,600)])
    text(0,-2220,'PLAN / W PAD 500 CHANGEABLE TO 400 OR 600 / CARRIER 900',45)
    # Front elevation; transverse y -> DXF x+4300.
    def front(pts,layer='FIXED'):poly([(4300+y,z) for y,z in pts],layer)
    for y in [-750,670]:front([(y,-1250),(y+80,-1250),(y+80,2050),(y,2050)])
    front([(-750,1970),(750,1970),(750,2050),(-750,2050)])
    front([(-40,1230),(40,1230),(40,1950),(-40,1950)])
    front([(-670,460),(670,460),(670,500),(-670,500)],'MOVING_LOW')
    for y in [-650,650]:line((4300+y,460),(4300+y,-640),'MOVING_LOW')
    front([(-670,-665),(670,-665),(670,-625),(-670,-625)],'MOVING_LOW')
    front([(-450,P['heel_low_z']-760),(450,P['heel_low_z']-760),(450,P['heel_low_z']),(-450,P['heel_low_z'])],'MOVING_LOW')
    line((4300,1230),(4300,480),'MOVING_LOW')
    line((4300,1950),(4300,1970))
    text(3470,2200,'FRONT / FRAME WIDTH 1500 / DATUM BELT Z=0',45)
    for code,val in [(0,'ENDSEC'),(0,'EOF')]:add(code,val)
    (OUT/'SF-01-R0-geometry.dxf').write_text('\n'.join(records)+'\n',encoding='ascii')

def write_docs():
    (OUT/'SF-01-R0-parameters.json').write_text(json.dumps(P,indent=2,ensure_ascii=False),encoding='utf-8')
    content='''# SF-01 R0 单级窄托台抬料器

状态：用于讨论与后续机构建模的尺寸方案，不是加工放行图。未修改现有主线，未运行新机构物料仿真。

## 文件
- PDF：两张 A3 横向图纸；100% 打印时各视图比例按图注明。
- SVG：同一绘图源生成的两张矢量图，可编辑文字与线条。
- DXF：三个分开的正投影视图，模型空间 1:1，单位 mm；英文注记以保证 CAD 兼容。无零件孔位或加工公差。
- parameters.json：全部候选基准尺寸；draw_concept.py：生成源。

## 坐标和尺寸闭合
X 为输送方向，Y 为横向，Z=0 为原接料主带带面。
托台后缘 X=1800，前缘 X=2100，进深300，中央顶面向下游倾30度。
低位鼻端 Z=-520；后缘 Z=-346.795；升程650后鼻端 Z=130、后缘 Z=303.205。
固定出料唇 X=2115、Z=100：高位鼻端比唇口高30，水平间隙15。
出料斜板从(2115,100)到(2315,0)，坡角26.565度，接原带面。
两侧斜肩向上游倾15度，低位后缘与中央后缘同高；前缘 Z=-266.410。
斜肩高位前缘约383.590；高位横梁底 Z=1110，名义净高约726。不能代替全姿态物料包络检查。
随动挡料裙板高760，低位下缘 Z=-1106.795，高位下缘 Z=-456.795；高位时仍比料盘前缘低110，防止余料进入抬料台下方开口。
门架顶 Z=2050，图示安装面 Z=-1250，概念包络高3300。实际带面离地高度未知，需核实；若现有空间不足，不能直接安装本版。
执行器下端 Z=1230，高位横梁连接点 Z=1130，留100名义杆端外露长度；执行器外形仍为占位，未指定产品。

## 重要边界
1. 来料仍为随机五块；图中煤块只说明空间与动作，没有设定真实轨迹或保证单块输出。
2. 900宽母托台配500宽中央承托面；可换400/500/600中央件，同时换相应斜肩，不做动态夹紧。
3. 靠重力回料依赖真实接触摩擦；煤湿、沾粉或棱角互锁时可能滞留。斜肩不保证排除并排或叠块。
4. 浅盘向入口收至920仍可能形成新的滞留或起拱位置；抬料能否拆开须单独验证。
5. 移动托台与料盘后缘、侧边、固定出料唇之间均存在运动接口。图中间隙只是几何占位，需专门设计柔性刮片、遮挡和可清理结构，不是防夹合格结论。
6. 上下限位、过载停止与防意外下落措施要在实体设计中落实。上行托住数块或卡料时，执行器负荷不能仅按一块煤的重量选型。
7. 下降前必须确认空台与接口无夹料；普通对射不等于单块计数。下游危险区未清空时不得开始下一次卸料。
8. 料盘底坡与回落会改变原先的单层分布；可能发生叠料、冲击与破碎。原模型不模拟破碎，不能据此作完整性结论。
9. 钢材牌号、型材规格、板厚、导轨额定载荷、焊缝、缸径、液压压力、紧固件及基础均未选定。
10. 当前模型“0.50m长轴上限”不是严格最大Feret尺寸；图纸不按0.50m认定最大空间包络。需要实际最大尺寸与质量数据。

## 建议验证顺序
先做几何扫掠和刚体受力估算，再做单块承托/卸料、最不利双块同托、随机五块完整循环。
分别记录单取、双取、空取、夹料、盘内剩料、掉落与破碎；不能用总通过率替代单块提取结果。

## 依据
现有接口：designs/main/config.json；粒度口径：README.md 与 singulator/lumps.py。
原理参考：https://performancefeeders.com/products/step-feeders
该资料面向小零件，描述先提一排再分选，仅支持原理来源，不支持本图尺寸、承载能力或单块成功率。
'''
    (OUT/'README.md').write_text(content,encoding='utf-8')

if __name__=='__main__':
    pdf=canvas.Canvas(str(OUT/'SF-01-R0-concept.pdf'),pagesize=(420*MM,297*MM),pageCompression=1)
    pdf.setTitle('SF-01 R0 单级窄托台抬料器 - 结构方案图')
    pdf.setAuthor('Coal singulator prototype / concept design')
    main_sheet(pdf);detail_sheet(pdf);pdf.save()
    dxf_export();write_docs()
    with zipfile.ZipFile(OUT/'SF-01-R0-editable.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.iterdir():
            if p.suffix in ['.svg','.dxf','.json','.md']:z.write(p,p.name)
        z.write(Path(__file__), 'draw_concept.py')
    print(OUT)
