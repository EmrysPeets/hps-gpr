from slide_helpers import *
A=W/'science/assets'

# Slide 3: existing instruction is replaced by the requested native table and two source plots.
delete('h5d7e580f29331fdc_0_262')
table(3,'selection',[
 ['Selection','2021 preselection + prompt sample'],
 ['Trigger / topology','Single-2 or Single-3; one selected good vertex'],
 ['Momenta [GeV]','p(track) > 0.4; p(e−) < 2.9; p(sum) > 2.8; p(vertex) < 4.0'],
 ['Track quality / hits','χ²(track)/ndf < 20; electron: 8 hits (updated selection)'],
 ['Vertex quality','No vertex-fit χ² requirement'],
 ['Positron cluster','E(cluster) > 0.2 GeV; inherited track–cluster timing cuts'],
],35,73,650,133,[135,515],13)
image(3,'momentum',A/'2021_preselection_electron_momentum.png',50,216,290,150)
image(3,'vertex',A/'2021_preselection_vertex_chi2.png',368,216,290,150)
text(3,'source','v5.0.5 preselection examples; these plots were not reprocessed with the updated eight-hit cut.',35,369,635,20,11)

# Slides 6/7: replace obsolete range screenshots with current, editable values and matching spectra.
rows=[['Run','Beam [GeV]','Sample','Search [MeV]','GP support [MeV]'],['2015','1.056','1170 nb⁻¹','19–100','14–135'],['2016','2.3','10608 nb⁻¹','39–180','30–210'],['2021','3.74','Native 10%','50–250','36–300']]
for n,ids in [(6,['g3f63fae4015_1_88','g3f63fae4015_1_87','g3f63fae4015_1_123']),(7,['g409df70e3d8_1_515','g409df70e3d8_1_519'])]:
    for oid in ids: delete(oid)
    if n==6:
        table(n,'ranges',rows,35,76,650,108,[65,90,170,150,175],14)
        text(n,'sample','2021 native 10% is drawn from an approximately 160 pb⁻¹ parent run.',36,185,645,24,13)
    else:
        table(n,'ranges',rows,26,77,428,119,[48,61,107,104,108],11.5)
        # Existing mixed-style box is removed as a whole mapped instruction object; reconstruct heading/body separately.
        delete('g409df70e3d8_1_523')
        text(n,'updates','2021 GP support begins at 36 MeV\n2015 search extends to 100 MeV',465,86,229,78,17,True)
        text(n,'distinction','Fit support supplies sidebands; the search range sets tested masses.',465,162,229,43,14)
    image(n,'spectra',A/'datasets_current_horizontal.png',27,216,667,160)
    text(n,'source','v5.0.5 archived spectra. Gray: GP support. Gold: tested search range.',35,377,620,18,10)

# Slide 9: preserve and reposition the existing RBF equation, with one plot for each requested parameter.
replace('h5d7e580f29331fdc_0_110','ℓ: correlation distance in log mass\nC: covariance amplitude\nα: diagonal bin-noise variance ≈ 1/y',335,78,342,74,17)
geom('h5d7e580f29331fdc_0_114',48,82,248,64.3)
image(9,'roles',A/'slide09_hyperparameter_roles.png',27,164,666,205)

# Slide 11: retain equations and flow; fill only the requested profile-plot slot.
delete('h5d7e580f29331fdc_0_388')
image(11,'profile78',A/'2021_profile_78MeV.png',463,82,238,278)
text(11,'legend','Gold: GP mean  |  Blue: profiled background\nRed: background + signal  |  Purple: signal',438,364,273,31,10)

# Slides 13/14: fit examples and explicitly conditional covariance-aware residual scan.
replace('h5d7e580f29331fdc_0_248','2021 native 10%: fixed GP predictions in three held-out windows',35,75,653,30,18)
image(13,'examples',A/'2021_fixed_state_examples.png',29,111,661,246)
text(13,'caption','Archived kernel states; ±2.25σ windows. Q/Nbin uses the full prediction covariance.',35,364,630,25,13)
replace('h5d7e580f29331fdc_0_255','Conditional residual size across the 2021 mass scan',35,75,655,30,18)
image(14,'q_equation',A/'slide14_residual_equation.png',52,109,610,41)
image(14,'residualscan',A/'2021_conditional_residual_scan.png',36,155,649,211)
text(14,'qualification','Nbin counts held-out bins, not fitted degrees of freedom. Correlated windows; no calibrated χ² or KS p-value.',35,367,640,28,12)

# Slide 21: paired source-specific threshold studies; keep diagnostic support distinct from production.
replace('h5d7e580f29331fdc_0_81','1% × 10: scale the fitted 1% mean; degree-five threshold continuation, with 40–300 MeV GP support.\nNative 10%: fit the saved 10% source; degree-six threshold continuation and 30–300 MeV support.',35,73,650,80,16,bullets=True)
image(21,'threshold',A/'slide21_threshold_comparison.png',28,165,666,201)
text(21,'qualification','Historical 65 MeV diagnostics. The selected production GP support is separately 36–300 MeV.',35,371,649,25,12)

save()
