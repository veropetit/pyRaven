
;+
; Visualization of the star
;
; :keyword lut: bla
; :keyword file_ps: bla
;-
pro visualisation, geom, ntheta, nphi, real_dtheta, real_dphi, toto, toto_txt=toto_txt, lut=lut, file_ps=file_ps

; This function is to generate figures of the fractionation of the stellar suface
; colour-coded for a certain value toto. 

	;print, min(toto), max(toto)

	if not keyword_set(lut) then lut = 3
	if not keyword_set(toto_txt) then toto_txt=''

	incl=geom[0]
	phase=geom[1]

	; creating vertice and polygon

	vert_theta = indgen(ntheta+1)*real_dtheta

	vert_phi = indgen(nphi[0])*real_dphi[0]
	ze_theta = [ replicate(vert_theta[0], nphi[0]), replicate(vert_theta[0+1], nphi[0]) ]
	ze_phi = [ vert_phi, vert_phi ]
	poly = lonarr(5, nphi[0])
	poly[0,*] = 4
	poly[1,*] = indgen(nphi[0])+0
	poly[2,*] = poly[1,*]+1
	poly[3,*] = poly[2,*]+nphi[0]-1
	poly[4,*] = poly[3,*]+1
	poly[2, nphi[0]-1] = poly[1 , 0]
	poly[4, nphi[0]-1] = poly[3 , 0]
	ze_poly =  poly
	i_vert = nphi[0]*2

	for i=1, ntheta-1 do begin
		vert_phi = indgen(nphi[i])*real_dphi[i]
		ze_theta = [ ze_theta,  replicate(vert_theta[i], nphi[i]), replicate(vert_theta[i+1], nphi[i]) ]
		ze_phi = [ ze_phi , vert_phi, vert_phi ]
		poly = lonarr(5, nphi[i])
		poly[0,*] = 4
		poly[1,*] = (indgen(nphi[i])+i_vert)
		poly[2,*] = poly[1,*]+1
		poly[4,*] = poly[2,*]+nphi[i]-1
		poly[3,*] = poly[4,*]+1
		poly[2, nphi[i]-1] = poly[1 , 0]
		poly[3, nphi[i]-1] = poly[4 , 0]	
		ze_poly = [ [ze_poly], [poly]]
		i_vert=i_vert + 2*nphi[i]
	endfor


	vertice =  fltarr(n_elements(ze_theta), 3)
	vertice[*,0] = sin(ze_theta)*cos(ze_phi)	;X ROT
	vertice[*,1] = sin(ze_theta) * sin(ze_phi)	;Y ROT
	vertice[*,2] = cos(ze_theta)				;Z ROT


	m_color=reform(ze_poly[1,*])
	vcolors=intarr(n_elements(ze_theta))*0
	;toto=A_los
	;toto_txt='A!ilos!n'
	vcolors[m_color]= bytscl(toto , min=min(toto), max=max(toto))


	oWindow = OBJ_NEW('IDLgrWindow', dimensions=[650,500], location=[40,0])  ; Create a window:  
	oScene = OBJ_NEW('IDLgrScene') ; Create a scene
	oView1 = OBJ_NEW('IDLgrView', dimension=[490,490], location=[5,5], color=[250,250,250])  ; Create a viewport
	oView2 = OBJ_NEW('IDLgrView', dimension=[145,490], location=[490,5], color=[250,250,250], VIEWPLANE_RECT = [0, 0, 1.0, 1.0])  ; Create a viewport

	oFont1 = OBJ_NEW('IDLgrFont', 'Hershey*6', size=9, thick=1.5)
	oFont2 = OBJ_NEW('IDLgrFont', 'Hershey*6', size=24, thick=1.5)
	oTitle = OBJ_NEW('IDLgrText', toto_txt, /enable_formatting , font=oFont2)

	oModel1 = OBJ_NEW('IDLgrModel')  ; Create a model (poly) for view1
	oModel2 = OBJ_NEW('IDLgrModel')  ; Create a model (colorbar) for view1
	oModel3 = OBJ_NEW('IDLgrModel')  ; Create a model (axis) for view1
	loadct,lut,/silent
	tvlct,rval,gval,bval,/get
	myPalette = OBJ_NEW('IDLgrPalette', rval, gval, bval)
	oColorbar = OBJ_NEW('IDLgrColorbar', PALETTE = myPalette, DIMENSIONS = [0.1, 0.8], /show_outline, threed=0, thick=2)  
	oAxis = OBJ_NEW('IDLgrAxis', direction=1,/exact, range=[min(toto),max(toto)], thick=2, title=oTitle, ticklen=0.1)
	oAxis->GetProperty, ticktext=blup
	blup->SetProperty, font=oFont1
	oAxis->SetProperty, ticktext=blub
	mypolygon = OBJ_NEW('IDLgrPolygon', transpose(vertice), polygons=ze_poly, VERT_COLORS=vcolors, palette=myPalette )

	;print,min(toto),max(toto)

	oScene->Add, oView1  ; add the view in the scene
	oScene->Add, oView2
	oView1->Add, oModel1
	oView2->Add, oModel2
	oView2->Add, oModel3
	oModel1->Add, mypolygon
	oModel2->Add, oColorbar
	oModel3->Add, oAxis
	;oModel1->Add, oTitle

	myplot1=OBJ_NEW('IDLgrPolygon',[0,0,0],[0,0,0],[0,1,1.5], style=1, thick=3, color=[0,0,255])
	myplot2=OBJ_NEW('IDLgrPolygon',[0,0,0],[0,1,1.5],[0,0,0], style=1, thick=3, color=[0,255,0])
	myplot3=OBJ_NEW('IDLgrPolygon',[0,1,1.5],[0,0,0],[0,0,0], style=1, thick=3, color=[255,0,0])
	myplot4=OBJ_NEW('IDLgrPolygon',[0,0,0],[0,0,0],[0,-1,-1.5], style=1, thick=3, color=[0,0,255], linestyle=1)
	myplot5=OBJ_NEW('IDLgrPolygon',[0,0,0],[0,-1,-1.5],[0,0,0], style=1, thick=3, color=[0,255,0], linestyle=1)
	myplot6=OBJ_NEW('IDLgrPolygon',[0,-1,-1.5],[0,0,0],[0,0,0], style=1, thick=3, color=[255,0,0], linestyle=1)

	oModel1->Add, myplot1
	oModel1->Add, myplot2
	oModel1->Add, myplot3
	oModel1->Add, myplot4
	oModel1->Add, myplot5
	oModel1->Add, myplot6

	oModel1->Scale, 0.65,0.65,0.65
	oModel1->Rotate, [-1,0,0],incl*180/!pi
	oModel1->Rotate, [0,sin(incl), cos(incl)], phase*180/!pi
	oModel2->Translate, 0.7, 0.1, 0
	scale = 0.8/(max(toto)-min(toto))
	;print, min(toto) * scale
	oModel3->Translate, 0.69, -1* min(toto)*scale +0.1, 0
	oModel3->Scale, 1.0, scale, 1.0, /premultiply
	mypolygon->SetProperty, STYLE=2, thick=4 ;,SHADING=1

	;SET_VIEW, oView2, oWindow
	;XOBJVIEW, oModel3

		oWindow->Draw, oScene  ; Draw the scene

	if keyword_set(file_ps) then begin
		oClip = OBJ_NEW('IDLgrClipboard') 
		oClip->Draw, oScene, VECTOR=1, POSTSCRIPT=1, FILENAME=file_ps 
	endif

	end


;+
; Returns the voigt_fara function
;-
function voigt_fara, u, a

; This function computes the H and L profiles. The real part of w is H and the imaginary part is L (Landi p.163, eqs 5.45)
; The implementation comes from http://adsabs.harvard.edu/abs/1982JQSRT..27..437H

	z = [complex(u,a)]
		;help, z
		;print, z[0:5]
	t_arr = complex(a,-1*u)
	s = [abs(u) + a]
	w = dcomplexarr(n_elements(u))

	for i=0,n_elements(u)-1 do begin
		t=t_arr[i]
	
	
		if (s[i] ge 15.0) then begin
			;region 1	
			;print,'région 1'
			w[i] = 0.5641896*t/(0.5 + t^2)
		endif else begin 
			if (s[i] ge 5.5) then begin
				;région 2
				;print,'région 2'
				w[i] = t*(1.410474 + 0.5641896*t^2)/((3.0 + t^2)*t^2 + 0.75)
			endif else begin
				if (imaginary(z[i]) ge (0.195*abs(real_part(z[i])) - 0.176) ) then begin
					;région 3
					;print,'région 3'
					w[i] = (16.4955 + t*(20.20933 + t*(11.96482 + t*(3.778987 + t*0.5642236))))/ $
						(16.4955 + t*(38.82363 + t*(39.27121 + t*(21.69274 + t*(6.699398 + t)))))
				endif else begin
					;région 4
					;print,'région 4'
					w[i] = exp(t^2) - t*(36183.31 - t^2* (3321.9905 - t^2* (1540.787 - t^2*(219.0313 - t^2* (35.76683 - t^2* (1.320522 - t^2*0.56419))))))/ $
						(32066.6 - t^2* (24322.84 - t^2*(9022.228 - t^2* (2186.181 - t^2* (364.2191 - t^2*(61.57037 - t^2* (1.841439 - t^2)))))))
				endelse
			endelse
		endelse
	endfor

return, w
end

;-------------------------------------------------------------------------------------------

function interpol_lin,  V, X, U
	;	V:	The input vector can be any type except string.
	;
	;	Irregular grids:
	;	X:	The absicissae values for V.  This vector must have same # of
	;		elements as V.  The values MUST be monotonically ascending
	;		or descending.
	;
	;	U:	The absicissae values for the result.  The result will have
	;		the same number of elements as U.  U does not need to be
	;		monotonic.  If U is outside the range of X, then the
	;		closest two endpoints of (X,V) are linearly extrapolated.

		m = N_elements(v)               ;# of input pnts

		s = VALUE_LOCATE(x, u) > 0L < (m-2) ;Subscript intervals.

		p = (u-x[s])*(v[s+1]-v[s])/(x[s+1] - x[s]) + v[s]


RETURN, p
end


;=========================================================================
; v2, 2012-11-16
function raven_zeeman, star_param_init, bconf, plot=plot, modulus=modulus, beff=beff, ngrid=ngrid, silent=silent ;, $
	;geff=geff, lambda0=lambda0, $ ; Line parameters
	;vsini=vsini, vdop=vdop,$ ; velocity field parameter
	;av=av, bnu=bnu, kappa=kappa,$ ; line transfert parameter 
	;Bpole=Bpole, incl=incl, beta=beta, phase=phase,$ ; field parameters
	;beff=beff, modulus=modulus,$; returned field average 
	;ndop=ndop, ngrid=ngrid, plot=plot,$ ; computational parameters
	;test=test ; undefined parameter to return some test values
		if keyword_set(silent) then silent=1 else silent=0
	star_param = star_param_init

;help, star_param, /struct

			; Input parameter in star_param
			; ==== Line parameters ===
			;	geff	- effective lande factor of the transition (triplet approximation)
			;	lambda0	- wave length of the transition in A
			; ==== Velocity field parameters ===
			; 	vsini	- projected velocity km/s 
			; 	vmac	- macroturbulence velocity (isotropic gaussian)
			;	vdop	- doppler broadening of the line 
			; ==== line transfert parameter ===
			;	logkappa	-  log of line to continuum opacity ratio
			;	av		- Voigt damping, unitless, Gamma'/Delta_nu_dop, see p.385 of Landi eq 9.24
			;	bnu		- Slope of the source function (1.5 for milnes-eddington atmosphere)
			;   ndop 	- number of spectral grid points per vdop (typically 50-100)
	
			; Input param in bconf
			; ==== Field parameters ===
			; 	Bpole	- polar field strength in Gauss default 0G
			;	incl	- inclination of the rotation axis [in rad]
			;	beta	- obliquity of the field pole to the rotational axis (in rad)
			;	phase	- rotational phase (in rad)

			; Other param and flags
			; ===== returned field average =====
			; 	beff	- brightness-averaged longitudinal field 
			;	modulus	- brightness-averaged field modulus
			; ==== computational parameters ====
			;	plot	- flag for visualisation

			; Important variables:
	
			;	real_n		- number of points in the grid
			;	ntheta		- number of annulus
			;	real_dtheta	- angular separation of the annulus
			;	nphi		- [ntheta] number of phase per annulus
			;	real_dphi	- [ntheta] angular separation of the phases
			;	grid_theta	- [real_n] theta values of the grid points
			;	grid_phi	- [real_n] phi values of the grid points
			;	A			- [real_n] Area of each grid points
	
			;	ROT			- [real_n, 3] XYZ position in ROT of each grid points
			;	LOS			- [real_n, 3] XYZ position in LOS of each grid points
			;	MAG			- [real_n, 3] XYZ position in MAG of each grid points
		
			;	vis			- [~real_n/2] index of visible grid points
			;	nvis		- number of visible elements	
			;	A_LOS		- [real_n] projected surface including limb darkening
			;	V_z			- [real_n] radial velocity (in vdop unit)
			;	B_MAG		- [real_n, 3] XYZ field value in MAG of each grid points
			;	B			- [real_n] Field strength of each grid point
			;	B_LOS		- [real_n] XYZ field value in LOS of each grid point


	
	
	; Create the dispersion grid					
	disp = raven_dispersion( star_param )
		if silent eq 0 then print,  '      number of spatial grid points: ', n_elements(disp.large)
				
	
		; Did an emperical scaling for the number of spatial grid points as a function of vsini / vdop
	if not keyword_set(ngrid) then begin

		if star_param.vsini le 50.0 then begin
		
			ngrid = 1000
			
		endif else begin
		
			if (star_param.vsini/star_param.vmac) le 8.0 then begin
				ngrid = 1000
			endif else begin
				if (star_param.vsini/star_param.vmac) ge 15 then begin
					print, '**** forcing vmac to ', star_param.vsini / 15.
					star_param.vmac = star_param.vsini / 15.
					ngrid = 5000
				endif else begin
					ngrid = fix(571*(star_param.vsini/star_param.vmac) - 3565)
				endelse
			endelse
			
		endelse


		;if star_param.vsini/star_param.vdop gt 1.0 then ngrid= fix(50* (star_param.vsini/star_param.vdop)^1.58) else ngrid=50
	endif
	if silent eq 0 then print, '      number of spatial grid points: ', ngrid


	;some constants
	const = { larmor : 1.3996e6, $ ; multiply by B(gauss) to get nu_B (in Hz)
				c : 2.99792458e5, $		; speed of light (km/s)
				a2cm : 1e-8, $
				km2cm : 1e5 $
			}


	value = { lorentz: star_param.bnu * star_param.geff / sqrt(!dpi) $
						* (star_param.line*const.a2cm) / (star_param.vdop*const.km2cm) * const.larmor $
			} ; needs to be multiplied by Bpole

	eps = 0.47


;==========
; Stuff to be calculated only once per star


	;*** Creation of the grid tied to the rotation frame

		; We want elements with roughtly the same area
		dA = (4*!pi)/ngrid

		dtheta = sqrt( 4*!pi/ngrid )
		ntheta = round( !pi/dtheta )	;the number of anulus is an integer
		real_dtheta = !dpi/ntheta	;the real dtheta
		theta = indgen( ntheta, /double ) * real_dtheta + real_dtheta/2.0	;array annulus angles
	
		dphi = 4*!pi / (ngrid * sin(theta) * real_dtheta )	;array of desired dphi for each annulus
		nphi = round( 2*!pi/dphi)		;number of phases per annulus (integer)
		real_dphi = 2*!dpi/nphi		;real dphi per annulus
		real_n = total(nphi)
	
		;create arrays with informations on each grid point
		grid_theta = replicate(theta[0], nphi[0])
		grid_phi = indgen(nphi[0],/double) * real_dphi[0] + real_dphi[0]/2

		;area of each grid point
		A = replicate( sin(theta[0])*real_dtheta*real_dphi[0], nphi[0] )
		for i=1, ntheta-1 do begin
			grid_theta = [ grid_theta, replicate(theta[i], nphi[i]) ]
			grid_phi = [ grid_phi, indgen(nphi[i],/float) * real_dphi[i] + real_dphi[i]/2 ]
			A = [ A, replicate( sin(theta[i])*real_dtheta*real_dphi[i], nphi[i] ) ]
		endfor
				;plot verification
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, grid_theta , toto_txt='Grid !7h!6'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, grid_phi , toto_txt='Grid !7u!6'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, A , toto_txt='A'

;return, 1

	;------------------------


	;------------------------
	;*** Compute the intensity line profile only once.


		w = voigt_fara(disp.large.u, star_param.av)
		voigt = real_part(w)
		fara = imaginary(w)


		
		profile_v_large = 10^star_param.logkappa * (star_param.av*fara - disp.large.u*voigt) / ( 1D + 10^star_param.logkappa * voigt/sqrt(!dpi) )^2

			; convolution at this level instead
			;profile_v_large = raven_convol_gauss( disp.large.wave, profile_v_large, star_param.vmac, star_param.resol)

			; Only for the disk-integrated i profile (see the comment on fac in the med_flux code
			if star_param.unno eq 0 then fac = (1D + 2.*star_param.bnu/3.) / (1.0 + star_param.bnu) else fac=1D
			
			profile_i_large = star_param.bnu / (1D + fac*10^star_param.logkappa * voigt/sqrt(!dpi) )
		
	

	;-----------------------
	;*** model structure
		;model = replicate( {wave:0D, vel:0D, u:0D, flux:0D, V:0D} , disp.nlarge )
		model = replicate( {wave:0D, vel:0D, u:0D, flux:0D, V:0D, flux2:0D, V2:0D, flux3:0D, flux4:0D} , disp.nlarge )
		model.u = disp.large.u
		model.vel = disp.large.vel + star_param.vrad
		model.wave = disp.large.wave
	


;==========
; Stuff to be re-calculated for each rotation (i-phase) change



	;*** Rotation matrix. 

		;rotation matrix LOS 2 ROT
		los2rot = [	[ cos(bconf.phase),		cos(bconf.incl)*sin(bconf.phase),	-1*sin(bconf.incl)*sin(bconf.phase)],$
						[ -1*sin(bconf.phase),	cos(bconf.incl)*cos(bconf.phase),	-1*sin(bconf.incl)*cos(bconf.phase)],$
						[ 0.0,				sin(bconf.incl),				cos(bconf.incl)] ]
		rot2los = transpose(los2rot)	;the invert of a rotation matrix is the transpose


	;-----------------
	; cartesian coordinates of each grid points in LOS, ROT

		;XYZ in ROT
		ROT = dblarr(real_n, 3)
		ROT[*,0] = sin(grid_theta) * cos(grid_phi)	;X ROT
		ROT[*,1] = sin(grid_theta) * sin(grid_phi)	;Y ROT
		ROT[*,2] = cos(grid_theta)					;Z ROT
				; plot verification
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, ROT[*,0] , toto_txt='X!iROT!n'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, ROT[*,1] , toto_txt='Y!iROT!n'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, ROT[*,2] , toto_txt='Z!iROT!n'


		;XYZ in LOS
		LOS = rot2los ## ROT
				; plot verification
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, LOS[*,0] , toto_txt='X!iLOS!n'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, LOS[*,1] , toto_txt='Y!iLOS!n'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, LOS[*,2] , toto_txt='Z!iLOS!n'

;return, 1

	;-----------------
	; Calulating values on the grid
	
		; visible elements where z component ge 0
		vis=where( LOS[*,2] ge 0.0, nvis )
	 
		;cos of angle between Zlos and the normal to the surface
		mu_los = LOS[*,2]
		Aneg = where(mu_los lt 0.0, acount)
		if acount gt 0 then mu_los[Aneg] = 0.0 ;just put all the unvisible values to zero
		; projected surface area in the LOS
		A_LOS = A * mu_los ;
		A_LOS /= total(A_LOS[vis]) ; Normalized to unity (fraction of the area spent by the region). 
	
		if star_param.unno eq 0 then A_LOS_V = (1. - eps + mu_LOS*eps) * A_LOS else A_LOS_V = mu_los * A_LOS
		if star_param.unno eq 0 then A_LOS_I = (1. - eps + mu_LOS*eps) * A_LOS else A_LOS_I = A_LOS

				; plot verification
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, mu_LOS , toto_txt='mu_LOS'
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, A_LOS  , toto_txt='Effective area'


		; radial velocity
		V_z = (star_param.vsini * LOS[*,0]) / star_param.vdop ; ==== in doppler units ====
				
				; plot verification
				;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, V_z , toto_txt='Radial velocity'
				;print, 'HERE!!!'
				;print, max(V_z)
				
				;hist = histogram(V_z[vis]*A_LOS*star_param.vdop, loc = loc)
				;plot, loc, hist, psym=10


;return, 1

;==========
; Stuff to be re-calculated for each field (beta) change



	;rotation matrix ROT 2 MAG
		rot2mag = [		[ 1,	0,				0],$
						[ 0,	cos(bconf.beta),		sin(bconf.beta)],$
						[ 0.0,	-1*sin(bconf.beta),	cos(bconf.beta)] ]
		mag2rot = transpose(rot2mag)	;the invert of a rotation matrix is the transpose

		
	;XYZ in B
		MAG = rot2mag ## ROT
				; plot verification
				;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, MAG[*,0] , toto_txt='X!iMAG!n'
				;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, MAG[*,1] , toto_txt='Y!iMAG!n'
				;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, MAG[*,2] , toto_txt='Z!iMAG!n'



	; This describe the "unity" vector field at the surface. It will need to be multiplied by the polar field strength.
	; but that is done only later other-wise the angle calculation will fail for the cas Bp=0. 
	; Field vectors on the surface
		; cos(thetab) = Zmag
		; sin(thetab) = sqrt( Xmag^2 + Ymag^2 )
		; cos(phib) = Xmag / sin(thetab)
		; sin(phib) = Ymag / sin(thetab)

		sinthetab = sqrt( MAG[*,0]^2 + MAG[*,1]^2 )
		no_null=where(sinthetab gt 0)
		B_MAG = dblarr(real_n, 3) *0
		B_MAG[no_null,0] = 3 * MAG[no_null,2] * sinthetab[no_null] * MAG[no_null,0]/sinthetab[no_null]	;Bx = 3cos(thetab)sin(thetab)cos(thetab)
		B_MAG[no_null,1] = 3 * MAG[no_null,2] * sinthetab[no_null] * MAG[no_null,1]/sinthetab[no_null]	;Bx = 3cos(thetab)sin(thetab)sin(thetab)
		B_MAG[*,2] = 3*MAG[*,2]^2-1.0	; Bz = 3cos(thetab)^2 - 1

			; plot verification
			;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, B_MAG[*,0] , toto_txt='B!iX!n (MAG)'
			;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, B_MAG[*,1] , toto_txt='B!iY!n (MAG)'
			;if keyword_set(plot) then visualisation, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, B_MAG[*,2] , toto_txt='B!iZ!n (MAG)'

	; B in LOS
		B_LOS = ( rot2los ##  ( mag2rot ## B_MAG ) )  ; ==== will need to be mutiply by Bpole/2 ===
	
			; plot verification
			;if keyword_set(plot) then visualisation, lut=1, [incl, phase], ntheta, nphi, real_dtheta, real_dphi, Bpole/2*B_LOS[*,2], toto_txt='B!iZ!n (LOS)'     
    
    
    ; Return the modulus and avg Bl (can be removed in the loop version)
    	;conti_int = ( 1.+ mu_LOS*star_param.bnu )
    	conti_int_area = A_LOS * ( 1.+ mu_LOS*star_param.bnu ) ; Note, A_LOS=0 if invisible
    	conti_flux = total(conti_int_area[vis])
    		;print, 'CONTI_FLUX ', conti_flux
    	B_modulus = sqrt( (bconf.Bpole/2.*B_MAG[*,0])^2 + (bconf.Bpole/2.*B_MAG[*,1])^2 + (bconf.Bpole/2.*B_MAG[*,2])^2 )
    	modulus = total( B_modulus * conti_int_area ) / conti_flux ; Note conti_int_area = 0 if invisible
		beff = total( (bconf.Bpole/2.*B_LOS[*,2]) * conti_int_area ) / conti_flux
			if silent eq 0 then print, '      Field modulus: ', modulus
			if silent eq 0 then print, '      Longitudinal field: ',beff

			; plot verification
    			;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, conti_int , toto_txt='conti flux'
    			;if keyword_set(plot) then visualisation, [bconf.incl, bconf.phase], ntheta, nphi, real_dtheta, real_dphi, conti_int_area , toto_txt='conti flux Area'

;return, 1 
     
     
;---------------------------
; Disk integration

	;---------
	; Loop over the visible elements
	for k=0L, nvis-1 do begin
		i = vis[k]
					
		prof_V = interpol_lin(profile_v_large, disp.large.u+V_z[i], model.u) ; Note, I put the mulos factor back in the area determination
		model.V += A_LOS_V[i] * B_LOS[i,2] * prof_V

		if star_param.unno eq 1 then begin
			prof_I = 1. + mu_los[i] * interpol(profile_i_large, disp.large.u+V_z[i], model.u)
		endif else begin
			prof_I = 1. + 1.0 * interpol(profile_i_large, disp.large.u+V_z[i], model.u)
		endelse
		model.flux += A_LOS_I[i]*prof_I

		;--------
		;prof_V = interpol_lin(profile_v_large, disp.large.u+V_z[i], model.u) * 1.0 * B_LOS[i,2] 
		;model.V2 += (1. - eps + mu_LOS[i]*eps) * A_LOS[i] * prof_V

		;prof_I = 1. + 1.0 * interpol(profile_i_large2, disp.large.u+V_z[i], model.u)
		;model.flux2 +=  (1. - eps + mu_LOS[i]*eps) * A_LOS[i] * prof_I
		;--------

	endfor
	
		
		model.V = model.V  * value.lorentz  / conti_flux
		;model.V2 = model.V2  * value.lorentz  / conti_flux

		model.V = raven_convol_gauss( model.wave, model.V, star_param.vmac, star_param.resol)
		;model.V2 = raven_convol_gauss( model.wave, model.V2, star_param.vmac, star_param.resol)

		model.V = model.V * bconf.Bpole		
		;model.V2 = model.V2 * bconf.Bpole		
		
			;model.flux = model.flux / (1 + 2./3.*star_param.bnu)
		model.flux = model.flux / conti_flux
		;model.flux2 = model.flux2 / conti_flux

		;model.flux3 = raven_med_flux(model.u, star_param.logkappa, av=star_param.av, bnu=star_param.bnu)
		;model.flux3 = raven_convol_vsini( model.wave, model.flux3, star_param.vsini )

		;model.flux4 = raven_med_flux_alt(model.u, star_param.logkappa, av=star_param.av, bnu=star_param.bnu)
		;model.flux4 = raven_convol_vsini( model.wave, model.flux4, star_param.vsini )


		model.flux = raven_convol_gauss( model.wave, model.flux, star_param.vmac, star_param.resol)
		;model.flux2 = raven_convol_gauss( model.wave, model.flux2, star_param.vmac, star_param.resol)
		;model.flux3 = raven_convol_gauss( model.wave, model.flux3, star_param.vmac, star_param.resol)
		;model.flux4 = raven_convol_gauss( model.wave, model.flux4, star_param.vmac, star_param.resol)


			; If one wants to check against zeeman_unno
			;model_test = zeeman_unno( down=[0,0,0], up=[0,1,1], $
			;				g_u=star_param.geff, lambda0=star_param.line, $ ; Line parameters
			;				vsini=star_param.vsini, vdop=star_param.vdop,$ 
			;				av=star_param.av, bnu=star_param.bnu, kappa=10^star_param.logkappa,$ ; line transfert parameter 
			;				Bpole=Bpole, incl=incl, beta=beta, phase=phase,$ ; field parameters
			;				beff=beff, modulus=modulus,$; returned field average 
			;				ndop=ndop, ngrid=ngrid )

				;;window, 2
				;;!p.multi = [0,1,2]
				;plot, model_test.vel, model_test.flux, /ynozero, yrange=[0.5,1.1];, xrange=[-5,5]
				;oplot, model.vel, model.flux, color=3, line=2;, /ynozero;, color=2, line=2
				;oplot, model.vel, flux, color=2, line=2
				;legend, ['Unno', 'Disk', 'conv'], color=[1,3,2], line=0, /bottom
			
				;;plot, model_test.vdop, model_test.V, xrange=[0,5];, yrange=[-0.0025, 0.0025]
				;;oplot, model.u, model.V, color=2, line=2
				;;!p.multi=0


return, model
end


