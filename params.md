
| Dict    | Key | diskint2 unno| diskint2 weak | loop | fit profile I |
| ----    | --- | -------- | ----| ---- | ------------- | 
| general   |lambda0    |x|x|x
|           |vsini      |x|x|x
|           |vdop       |x|x|x
|           |av         |x|x|x
|           |bnu        |x|x|x
|           |logkappa   |x|x|x
|           |ndop       |x|x|x
|           |Bpole      |x|x| 
|           |incl       |x|x| 
|           |beta       |x|x| 
|           |phase      |x|x| 
|unnoparam  |down       |x| | 
|           |up         |x| | 
|weakparam  |geff       | |x|x
|gridparam  |Bgrid      | | |x
|           |igrid      | | |x
|           |betagrid   | | |x
|           |phasegrid  | | |x

A description of the indiviudal keys are given below.
 * lambda0 - the central wavelength of the transition
 * vsini - the rotational velocity along our line of sight
 * vdop - the thermal broadening
 * av - the damping coefficient of the Voigt profile
 * bnu - the slope of the source function with respect to the vertical optical depth
 * logkappa - the log of the line strength parameter
 * ndop - the number of sample points per doppler width for the wavelength array
 * Bpole - the dipoler magnetic field strength
 * incl - the inclination of the star's rotation axis with respect to our line of sight
 * beta - the obliquity, or the tilt of the star's magnetic axis with respect to the star's rotation axis
 * phase - the rotational phase of the star
 * down - the s, j, and l of the lower level energy split
 * up - the s, j, and l of the upper lever energy split
 * geff - effective Lande factor
 * Bgrid - the grid of possible Bpole values
 * igrid - the grid of possible inclination values
 * betagrid - the grid of possible beta values
 * phasegrid - the grid of possible phase values