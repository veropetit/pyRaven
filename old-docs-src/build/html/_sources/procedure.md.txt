# Procedure

## Control files

### Catalog_flag

The main control file is called `catalog_flag.dat`. It will tell the main code `raven_main` which task to execute for which star in the sample. 
* Setting the flag to 0 do nothing
* Setting the flag to 1 will run the associated task (and the code will switch it to 2)
* The 2 flag indicates the task was done (change back to 1 to run it again)
The only exeption to the above is RZEE (see below)

The ID numbers can be anything -- for example in the MiMeS O-star sample, the single star sample started at 1, and the binary star sample was handled separately with control file that started at 101 (so that each star in the combined sample has a unique identifyer). 


    ID          ROOT         SDATA         RDATA          SFIT          RFIT          RZEE          RPDF          RREG          PPDF         RODDS         RGEOM         PGEOM
    0             1             0             0             0             0             0             0             0             0             0             0             0
    1       hd13745             0             0             0             0             0             0             0             0             0             0             0
    2       hd14633             0             0             0             0             0             0             0             0             0             0             0



### build_catalog.pro

This program lives in the working directory and its parameters are edited for each project.

It is also possible to adjust this procedure to facilitate reading in the necessary information from files. 

The parameters for each stars are stored in a structure. The build_catalog program defines a template structure (with common default values), and then uses user-defined commands to create a folder and a star_param structure for each star in the sample. 

The parameters are:

* id: integer, ID number for the star, 
* root: string, root name to be used (could be the name of the star, for example),
* data_dat:'data_list.dat', 
* folder: string, name of the folder for the star 
* nobs: integer, number of observations for the star
* geff:1.2, line:5000., wint:0.05D, 
* wint_lsd:0.1, wpol_lsd:60, $ ; note, wpol in nm
* vsini:0D, vmac:0D, vrad:0D, vdop:7D, $ 
* logkappa:0D, av:0.01D, bnu:1.5, ndop:15, $
* resol:65000D, unno:0, bin:1, $
* int_flag:[ !values.f_nan, !values.f_nan, !values.f_nan, !values.f_nan ], $ ;[logkappa, vmac, vsini, vrad]
* B_max:0., v_max:700, $
* f_want:[ 0.997, 0.99, 0.954, 0.683 ] $
            }                    



