import sys
sys.path.insert(0, "src")
import numpy as np
from stewbench.reporting.style import apply_style, FAMILY_COLOR, FAMILY_MARKER, TEXT_SECONDARY, TEXT_PRIMARY, GRID
apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
OUT = sys.argv[1]

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# ---------------- Figure 1: benchmark
bench = [
 ("Spectral + LR","Feature-based",.829,0,.796,.861),("Spectral + SVM","Feature-based",.829,0,.791,.865),
 ("Spectral + sLDA","Feature-based",.827,0,.795,.858),("Riemann re-centred TS + LR†","Riemannian",.825,0,.778,.865),
 ("Spectral + RF","Feature-based",.821,.002,.786,.854),("Spectral + GBDT","Feature-based",.805,0,.769,.840),
 ("ATCNet","EEG Transformer",.803,.010,.764,.841),("EEGNet-8,2","EEG CNN",.801,.014,.763,.839),
 ("EEG-TCNet","EEG CNN",.797,.004,.757,.836),("Riemann TS + LR","Riemannian",.795,0,.757,.831),
 ("TSception","EEG CNN",.794,.006,.756,.830),("ShallowConvNet","EEG CNN",.785,.012,.750,.819),
 ("1D-CNN","Generic CNN",.784,.004,.747,.819),("EEG Conformer","EEG Transformer",.778,.005,.739,.815),
 ("CNN-LSTM","Generic CNN",.776,.012,.742,.811),("GRU","Recurrent",.766,.005,.730,.801),
 ("BiLSTM","Recurrent",.765,.005,.731,.798),("LSTM","Recurrent",.763,.014,.732,.792),
 ("DGCNN","Graph",.761,.010,.725,.796),("Swin (STFT)","Vision Transformer",.740,.013,.706,.774),
 ("ViT (STFT)","Vision Transformer",.734,.008,.699,.768),("Electrode GCN","Graph",.724,.009,.685,.762),
 ("Riemann MDM","Riemannian",.713,0,.665,.762),("Vanilla RNN","Recurrent",.703,.009,.677,.730),
 ("DeepConvNet","EEG CNN",.633,.087,.591,.678)]
fig, ax = plt.subplots(figsize=(6.6, 7.4))
for i,(n,f,m,sd,lo,hi) in enumerate(bench[::-1]):
    ax.plot([lo,hi],[i,i],color=FAMILY_COLOR[f],lw=1.6,alpha=.55)
    ax.plot(m,i,marker=FAMILY_MARKER[f],color=FAMILY_COLOR[f],ms=6.5,mec="white",mew=.8,ls="none")
    ax.text(hi+.006,i,f"{m:.3f}",va="center",fontsize=6.5,color=TEXT_SECONDARY)
ax.set_yticks(range(len(bench)),[b[0] for b in bench[::-1]],fontsize=7)
ax.axvline(.5,color=TEXT_SECONDARY,ls=":",lw=1)
ax.set_xlim(.55,.92); ax.set_xlabel("Balanced accuracy on held-out subjects (95% subject-bootstrap CI)")
ax.grid(axis="y",visible=False)
fams=[f for f in FAMILY_COLOR if any(b[1]==f for b in bench)]
ax.legend(handles=[Line2D([],[],marker=FAMILY_MARKER[f],color=FAMILY_COLOR[f],ls="none",ms=6,label=f) for f in fams],
          loc="lower right",fontsize=6.5,frameon=True,framealpha=.95,edgecolor=GRID)
save(fig,"fig1_benchmark")

# ---------------- Figure 2: artifact ablation
models=["Spectral\n+ SVM","Spectral\n+ LR","Spectral\n+ sLDA","Riemann\nTS + LR","ATCNet","EEGNet","EEG-TCNet","TSception","Shallow\nConvNet"]
main=[.829,.829,.827,.795,.803,.801,.797,.794,.785]; nog=[.825,.824,.819,.799,.797,.795,.789,.805,.786]
noft=[.778,.777,.770,.776,.783,.780,.775,.780,.768]; both=[.785,.776,.769,.777,.777,.770,.769,.770,.759]
x=np.arange(len(models)); w=.2
fig,ax=plt.subplots(figsize=(7.2,3.2))
for k,(vals,lab,c) in enumerate([(main,"All channels, 0.5–45 Hz","#2a78d6"),(nog,"Gamma removed (0.5–30 Hz)","#1baf7a"),
                                  (noft,"F7/F8/T7/T8 zeroed","#eb6834"),(both,"Both removed","#4a3aa7")]):
    ax.bar(x+(k-1.5)*w,vals,w*.92,color=c,label=lab,edgecolor="white",linewidth=.6)
ax.set_xticks(x,models,fontsize=7); ax.set_ylim(.70,.85); ax.set_ylabel("Balanced accuracy")
ax.legend(ncol=4,fontsize=6.5,loc="upper center",bbox_to_anchor=(.5,1.14))
ax.grid(axis="x",visible=False)
save(fig,"fig2_ablation")

# ---------------- Figure 3: data efficiency
n=[6,12,24,37]
curves={
 "EEG Conformer":{"none":[.632,.668,.773,.776],"CWGAN-GP (100%)":[.661,.752,.764,.782],"DDPM (100%)":[.641,.733,.773,.780],"Channel dropout":[.657,.674,.778,.801]},
 "EEGNet-8,2":{"none":[.712,.742,.775,.801],"CWGAN-GP (100%)":[.710,.727,.738,.768],"DDPM (100%)":[.703,.727,.763,.790],"Channel dropout":[.717,.745,.780,.807]},
 "ShallowConvNet":{"none":[.733,.743,.774,.794],"CWGAN-GP (100%)":[.735,.756,.777,.769],"DDPM (100%)":[.726,.772,.780,.799],"Channel dropout":[.745,.752,.788,.810]}}
cols={"none":TEXT_SECONDARY,"CWGAN-GP (100%)":"#eb6834","DDPM (100%)":"#1baf7a","Channel dropout":"#2a78d6"}
mk={"none":"o","CWGAN-GP (100%)":"s","DDPM (100%)":"^","Channel dropout":"D"}
fig,axes=plt.subplots(1,3,figsize=(7.4,2.7),sharey=True)
for ax,(m,d) in zip(axes,curves.items()):
    for k,v in d.items():
        ax.plot(n,v,marker=mk[k],color=cols[k],ls="--" if k=="none" else "-",lw=1.6,ms=4,label=k)
    ax.set_title(m,fontsize=8); ax.set_xticks(n,["6","12","24","all\n(~37)"]); ax.set_xlabel("Training subjects per fold",fontsize=7)
axes[0].set_ylabel("Balanced accuracy")
axes[0].legend(fontsize=6,loc="lower right")
fig.tight_layout(); save(fig,"fig3_data_efficiency")

# ---------------- Figure 4: Model E architecture (diagram)
fig,ax=plt.subplots(figsize=(7.6,4.0)); ax.set_xlim(0,104); ax.set_ylim(0,54); ax.axis("off")
def box(x,y,w,h,text,fc,ec,fs=6.6,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.4,rounding_size=1.2",fc=fc,ec=ec,lw=1.1))
    ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=fs,color=TEXT_PRIMARY,fontweight="bold" if bold else "normal")
def arrow(x1,y1,x2,y2,rad=0.0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=9,color=TEXT_SECONDARY,lw=1.1,
                                 connectionstyle=f"arc3,rad={rad}"))
blue,blue_e="#e9f2fc","#2a78d6"; org,org_e="#fdeee7","#eb6834"; grn,grn_e="#e6f6f0","#1baf7a"; gry,gry_e="#f4f4f2","#8a8984"
box(1,35,19,12,"Training windows\n(inner-training\nsubjects,\nboth classes)",gry,gry_e)
box(26,41,23,9,"Artifact bank\nocular topographies,\nreal waveforms, EMG profile",org,org_e)
box(26,27,23,9,"Counterfactual generator\nadd / remove ocular,\nadd EMG, re-standardise",org,org_e)
box(1,16,19,9,"Channel dropout\n(p = 0.5)",gry,gry_e)
box(55,36,15,8,"Original\nwindow  x",blue,blue_e)
box(55,20,15,8,"Counterfactual\nwindow  x\u0303",blue,blue_e)
box(75,26,15,12,"Shared encoder\n(EEGNet, ATCNet\nor SpecNet)",grn,grn_e)
box(94,37,9,6,"p(y | x)",blue,blue_e,fs=6.2)
box(94,21,9,6,"p(y | x\u0303)",blue,blue_e,fs=6.2)
arrow(20,44,26,45); arrow(20,36.5,26,31.5); arrow(37.5,41,37.5,36)
arrow(10.5,35,10.5,25); arrow(20,20.5,55,23)
arrow(49,31.5,55,25); arrow(20,38.6,55,38.6)
arrow(70,40,75,34); arrow(70,24,75,30)
arrow(90,35,94,40); arrow(90,29,94,24)
box(18,2,72,9,"Loss  =  \u00bd CE(x, y)  +  \u00bd CE(x\u0303, y)  +  \u03bb \u00b7 SymKL( p(y|x) \u2016 p(y|x\u0303) ),   \u03bb = 3",gry,gry_e,fs=7,bold=True)
arrow(98.5,21,90.5,8)
arrow(103.3,40,90.5,9.5,rad=-0.35)
ax.text(37.5,52.6,"estimated per fold from training subjects only; independent of the class label",ha="center",fontsize=6,color=TEXT_SECONDARY,style="italic")
save(fig,"fig4_model_e")

# ---------------- Figure 5: Model E dev round 2
conds=["Plain\n(ERM)","Channel\ndropout","ACCT\n(λ=1)","ACCT\n(λ=3)","ACCT + CD\n(λ=1)","ACCT + CD\n(λ=3)"]
clean=[.768,.778,.780,.790,.792,.807]; abl=[.630,.683,.663,.673,.711,.710]; flips=[.245,.167,.134,.106,.078,.070]
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.4,2.8),gridspec_kw={"width_ratios":[1.5,1]})
x=np.arange(len(conds))
a1.bar(x-.19,clean,.36,color="#2a78d6",label="Clean windows",edgecolor="white")
a1.bar(x+.19,abl,.36,color="#eb6834",label="F7/F8/T7/T8 zeroed at test",edgecolor="white")
for i,(c,b) in enumerate(zip(clean,abl)):
    a1.text(i-.19,c+.004,f"{c:.3f}",ha="center",fontsize=5.5,color=TEXT_SECONDARY)
    a1.text(i+.19,b+.004,f"{b:.3f}",ha="center",fontsize=5.5,color=TEXT_SECONDARY)
a1.set_xticks(x,conds,fontsize=6); a1.set_ylim(.55,.84); a1.set_ylabel("Balanced accuracy"); a1.legend(fontsize=6,loc="upper left")
a1.grid(axis="x",visible=False)
a2.bar(x,[f*100 for f in flips],.6,color="#4a3aa7",edgecolor="white")
for i,f in enumerate(flips): a2.text(i,f*100+.5,f"{f*100:.1f}%",ha="center",fontsize=6,color=TEXT_SECONDARY)
a2.set_xticks(x,["ERM","CD","ACCT λ=1","ACCT λ=3","ACCT+CD λ=1","ACCT+CD λ=3"],fontsize=6,rotation=35,ha="right"); a2.set_ylabel("Predictions flipped (%)"); a2.set_title("Injected artifacts, strength ×3",fontsize=7.5)
a2.grid(axis="x",visible=False)
fig.tight_layout(); save(fig,"fig5_model_e_dev")
print("ok")
