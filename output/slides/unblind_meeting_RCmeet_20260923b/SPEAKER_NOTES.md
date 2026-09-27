# Short explanations for the latest revisions

The [three slide-9 scripts](../unblind_meeting_RCmeet_20260923/SPEAKER_SCRIPTS.md) remain available. The slide-29 walkthrough below supersedes that earlier document's coupling walkthrough.

## Slide 13

“Each lower panel shows the standardized residual in each mass bin. We subtract the GP prediction from the data and divide by the combined counting and GP uncertainty. Zero means agreement, and a positive value means more data than predicted. The green interval marks plus or minus two estimated standard deviations. These residuals share overlapping GP fits, so neighboring bins are correlated.”

## Slide 14

“At each point on the horizontal axis we leave out a different ±2.25-sigma window and fit the remaining sidebands. D_side adds the Poisson discrepancy between each scored sideband count and the fitted GP mean. A bin contributes zero when data equal the prediction. Dividing by the number of sideband bins gives the ratio on the vertical axis.

“We judge the value against toys that repeat this procedure. The dashed line is their median, around 0.95, and the blue band contains the central 90 percent at each center. All displayed data values lie within those bands. One is not an exact target because fitting changes the residuals. These are correlated, conditional checks of the fitted sidebands; predicting the omitted region requires separate validation.”

## Slide 29

“The BEST relation connects the narrow signal yield to the radiative-trident background at the same mass. We insert the 90-percent yield limit from the preceding slide, divide by the local radiative-background density and mass, and apply the known factors shown here. N_f is the inverse electron branching fraction, which accounts for other open decay channels. That correction is already present in the curves. The box uses the density form of BEST equation 19, in the same yield notation as my earlier presentation.”

Sources, definitions and technical qualifications are saved in the native speaker notes and the [changelog](CHANGELOG.md).
