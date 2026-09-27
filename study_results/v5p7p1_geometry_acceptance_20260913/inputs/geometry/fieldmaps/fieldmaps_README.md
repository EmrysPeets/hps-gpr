# hps-fieldmaps

These are the fieldmaps for both SLIC (or hps-sim) detector modeles and for 
hps-java reconstruction. They are created from the text files stored in Confluence in the
[Beamline and Magnet](https://confluence.slac.stanford.edu/display/hpsg/Beamline+and+Magnet) 
page, which also shows the procedure for producing the corrected fieldmaps.
The map name should be:  
<code>
<current-density>acm2-<original field strenght>-corrected-unfolded-scaled-<scale factor>.dat
</code>

However, one can observe that the "current-density" number is _very_ approximate. The field files will be
compressed with the command "tar czf name.tar.gz name.dat"

For details on how to produce new fieldmaps see: 
[Magnetic_Field_Calculations.ipynb](https://github.com/JeffersonLab/hps-fieldmaps/blob/master/Magnetic_Field_Calculations.ipynb)

Index of available fields maps:

|                                     *filename* | *Beam energy* |     *run period* |
|-----------------------------------------------:|--------------:|-----------------:|
|   125acm2_3kg_corrected_unfolded_scaled_0.7992 |     1.056 GeV | Engineering 2015 | 
| 209acm2_5kg_corrected_unfolded_scaled__1.04545 |     2.306 GeV | Physics Run 2016 |
|  418acm2_10kg_corrected_unfolded_scaled_1.0319 |      4.55 GeV | Physics Run 2019 |
|  334acm3_8kg_corrected_unfolded_scaled_1.07326 |     3.742 GeV | Physics Run 2021 |
 |  166acm2_4kg_corrected_unfolded_scaled_1.10179 |      1.92 GeV | Physics Run 2021 |   


|                                     *filename* | *Beam energy* | *Proposal Detector* |
|-----------------------------------------------:|--------------:|--------------------:|
|   125acm2_3kg_corrected_unfolded_scaled_0.7992 |      1.05 GeV | Proposal 2017 1pt05 | 
| 166acm2_4kg_corrected_unfolded_scaled_1.090896 |      1.92 GeV |  Proposal 2021 1pt9 |
| 209acm2_5kg_corrected_unfolded_scaled__1.04545 |       2.3 GeV |  Proposal 2017 2pt3 |
|  418acm2_10kg_corrected_unfolded_scaled_0.9979 |       4.4 GeV |  Proposal 2017 4pt4 |
|   334acm3_8kg_corrected_unfolded_scaled_1.0508 |       3.7 GeV |  Proposal 2021 3pt7 |
|  627acm2_13kg_corrected_unfolded_scaled_1.1538 |       6.6 GeV |  Proposal 2017 6pt6 |


