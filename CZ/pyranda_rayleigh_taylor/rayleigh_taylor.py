import argparse
import sys
import textwrap
import numpy
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep


# Setup some defaults
DEFAULTS = {
    "tstop": 100.0,
    "stop-width-fraction": 0.5,
    "max_iter": 10000
}

def run_sim(args):
    Npts = args.num_points
    dim = args.dim

    if dim == "2D":
        is2D = True
    elif dim == "3D":
        is2D = False
    else:
        raise ValueError(f"Unknown dim value '{dim}'")

    if 'seed' in args:
        random_ic_seed = args.seed
    else:
        random_ic_seed = None

    headless = args.headless

    print(args)
    if not args.tstop:
        print("tstop not specified")
    if not args.stop_width_fraction:
        print("stop-width-fraction not specified")
    ## Define a mesh
    if is2D:
        widthToHeight = 4.0
        xwidth = widthToHeight*numpy.pi
        problem = "RAYLEIGH_TAYLOR_2D"
        imesh = (
            """
        xdom = (-xwidth/2.0, xwidth/2.0 , int(Npts*widthToHeight), periodic=False)
        ydom = (0.0, 2*pi*FF,  Npts, periodic=True)
        zdom = (0.0, 2*pi*FF,  1, periodic=True)
        """.replace(
                "Npts", str(Npts)
            )
            .replace("pi", str(numpy.pi))
            .replace("FF", str(float(Npts - 1) / Npts))
            .replace("xwidth", str(xwidth))
            .replace("widthToHeight",str(widthToHeight))
        )
        waveLength = 4
    else:
        widthToHeight = 3.0
        xwidth = widthToHeight*numpy.pi
        problem = "RAYLEIGH_TAYLOR_3D"
        imesh = (
            """
        xdom = (-xwidth/2.0, xwidth/2.0, int(Npts*widthToHeigth), periodic=False)
        ydom = (0.0, 2*pi*FF,  Npts, periodic=True)
        zdom = (0.0, 2*pi*FF,  Npts, periodic=True)
        """.replace(
                "Npts", str(Npts)
            )
            .replace("pi", str(numpy.pi))
            .replace("FF", str(float(Npts - 1) / Npts))
            .replace("xwidth", str(xwidth))
        )
        waveLength = 4

    # Read in fluid properties
    if numpy.abs(args.atwood_number - 1.0) < 1.e-6:
        raise ValueError("Atwood number cannot be exactly 1: max is 1.0 - 1.e-6")

    rho_l = args.light_density
    rho_h = rho_l*(args.atwood_number + 1.0)/(1.0 - args.atwood_number) # NOTE: singularity here: safer to set via rho_h + atwood?  Would fluid over vacuum scenario even be handeld by the miranda model below?
    print(f"{args.atwood_number=}")
    print(f"{rho_l=}, {rho_h=}")
    # rho_l = 1.0  # Density of light fluid
    # rho_h = 3.0  # Density of heavy fluid
    mwH = rho_h # 3.0  # Molar masses of heavy/light fluids?
    mwL = rho_l # 1.0  # NOTE: should these really be the same as density? -> likely only for split exactly in half
    gx = (
        -0.01
    )  # NOTE: is this gravity?  if so lets switch to y to cutout using vist transforms to rotate
    Runiv = 1.0  # Universal gas constant from ideal gas law: pV=nRT
    CPh = 1.4    # Constant pressure specific heat of heavy gas
    CVh = 1.0    # Constant volume specific heat of heavy gas
    CPl = 1.4    # Constant pressure specific heat of light gas
    CVl = 1.0    # Constant volume specific heat of light gas

    # Initial conditions
    ranVelMag = args.random_velocity_magnitude
    velMag    = args.velocity_magnitude
    velMod    = args.velocity_modes
    velThick  = args.velocity_thickness
    
    parm_dict = {
        "gx": gx,
        "CPh": CPh,
        "CPl": CPl,
        "CVh": CVh,
        "CVl": CVl,
        "mwH": mwH,
        "mwL": mwL,
        "Runiv": Runiv,
        "waveLength": waveLength,
        "rho_l": rho_l,
        "rho_h": rho_h,
        "delta": 2.0 * numpy.pi / Npts * velThick,
        "ranVelMag": ranVelMag,
        "velMag": velMag,
        "velMod": velMod,
    }

    # Initialize a simulation object on a mesh
    ss = pyrandaSim(problem, textwrap.dedent(imesh))
    ss.addPackage(pyrandaTimestep(ss))
    ss.addPackage(pyrandaBC(ss))

    # User defined function to take mean of data and return it as 3d field, dataBar
    def meanXto3d(pysim, data):
        meanX = pysim.PyMPI.yzsum(data) / (pysim.PyMPI.ny * pysim.PyMPI.nz)
        tmp = pysim.emptyScalar()
        for i in range(tmp.shape[0]):
            ii = int(pysim.mesh.indices[0].data[i, 0, 0])
            tmp[i, :, :] = meanX[ii]
        return tmp

    ss.addUserDefinedFunction("xbar", meanXto3d)

    def getPressure(pysim,p0):
        dx       = pysim.dx
        rho1d    = pysim.var('rho')[:,0,0]    
        pressure = pysim.emptyScalar()
        meshx    = pysim.mesh.coords[0].data
        pbot     = p0 - numpy.trapz(rho1d*gx)*dx
        
        for i in range(meshx.shape[0]):
            i_local = int(pysim.mesh.indices[0].data[i,0,0])
            p_i     = pbot + numpy.trapz( rho1d[:i_local] * gx )*dx
            pressure[i,:,:] = p_i
    
        return pressure

    ss.addUserDefinedFunction("getPressure",getPressure)

    
    # Define the equations of motion
    eom = """
    # Primary Equations of motion here
    ddt(:rhoYh:)  =  -ddx(:rhoYh:*:u: - :Jx:)    - ddy(:rhoYh:*:v: - :Jy:)   - ddz(:rhoYh:*:w: - :Jz:)
    ddt(:rhoYl:)  =  -ddx(:rhoYl:*:u: + :Jx:)    - ddy(:rhoYl:*:v: + :Jy:)   - ddz(:rhoYl:*:w: + :Jz:)
    ddt(:rhou:)   =  -ddx(:rhou:*:u: - :tauxx:)  - ddy(:rhou:*:v: - :tauxy:) - ddz(:rhou:*:w: - :tauxz:) + :rho:*gx
    ddt(:rhov:)   =  -ddx(:rhov:*:u: - :tauxy:)  - ddy(:rhov:*:v: - :tauyy:) - ddz(:rhov:*:w: - :tauyz:)
    ddt(:rhow:)   =  -ddx(:rhow:*:u: - :tauxz:)  - ddy(:rhow:*:v: - :tauyz:) - ddz(:rhow:*:w: - :tauzz:)
    ddt(:Et:)     =  -ddx( (:Et: - :tauxx:)*:u: - :tauxy:*:v: - :tauxz:*:w: - :tx:*:kappa:) - ddy( (:Et: - :tauyy:)*:v: -:tauxy:*:u: - :tauyz:*:w: - :ty:*:kappa:) - ddz( (:Et: - :tauzz:)*:w: - :tauxz:*:u: - :tauyz:*:v: - :tz:*:kappa:) + :rho:*gx*:u:
    # Conservative filter of the EoM
    :rhoYh:     =  fbar( :rhoYh:  )
    :rhoYl:     =  fbar( :rhoYl:  )
    :rhou:      =  fbar( :rhou: )
    :rhov:      =  fbar( :rhov: )
    :rhow:      =  fbar( :rhow: )
    :Et:        =  fbar( :Et:   )
    # Xbar operator
    :mybar: = xbar(:rho:*:u:*:u:)
    # Update the primatives and enforce the EOS
    :rho:       = :rhoYh: + :rhoYl:
    :Yh:        =  :rhoYh: / :rho:
    :Yl:        =  :rhoYl: / :rho:
    :u:         =  :rhou: / :rho:
    :v:         =  :rhov: / :rho:
    :w:         =  :rhow: / :rho:
    :cv:        = :Yh:*CVh + :Yl:*CVl
    :cp:        = :Yh:*CPh + :Yl:*CPl
    :gamma:     = :cp:/:cv:
    :p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
    :mw:        = 1.0 / ( :Yh: / mwH + :Yl: / mwL )
    :R:         = Runiv / :mw:
    :T:         = :p: / (:rho: * :R: )
    # Artificial bulk viscosity / shear viscosity
    :ux:        =  ddx(:u:)
    :vy:        =  ddy(:v:)
    :wz:        =  ddz(:w:)
    :div:       =  :ux: + :vy: + :wz:
    # Remaining cross derivatives
    :uy:        =  ddy(:u:)
    :uz:        =  ddz(:u:)
    :vx:        =  ddx(:v:)
    :vz:        =  ddz(:v:)
    :wy:        =  ddy(:w:)
    :wx:        =  ddx(:w:)
    :Yx:        =  ddx(:Yh:)
    :Yy:        =  ddy(:Yh:)
    :Yz:        =  ddz(:Yh:)
    :enst:      = sqrt( (:uy:-:vx:)**2 + (:uz: - :wx:)**2 + (:vz:-:wy:)**2 )
    :S:         = sqrt( :ux:*:ux: + :vy:*:vy: + :wz:*:wz: + .5*((:uy:+:vx:)**2 + (:uz: + :wx:)**2 + (:vz:+:wy:)**2) )
    :mu:        = 1.0e-4 * gbar( abs(ring(:S:  )) ) * :rho:
    :beta:      = 7.0e-2 * gbar( abs(ring(:div:)) * :rho: )
    # Artificial species diffusivities
    :Dsgs:      =  1.0e-4 * ring(:Yh:)
    :Ysgs:      =  1.0e2  * (abs(:Yh:) - 1.0 + abs(1.0-:Yh: ) )*gridLen**2
    :adiff:     =  gbar( :rho:*numpy.maximum(:Dsgs:,:Ysgs:) / :dt: )
    :Jx:        =  :adiff:*:Yx:
    :Jy:        =  :adiff:*:Yy:
    :Jz:        =  :adiff:*:Yz:
    :taudia:    =  (:beta:-2./3.*:mu:) *:div: - :p:
    :tauxx:     =  2.0*:mu:*:ux:   + :taudia:
    :tauyy:     =  2.0*:mu:*:vy:   + :taudia:
    :tauzz:     =  2.0*:mu:*:wz:   + :taudia:
    :tauxy:     = :mu:*(:uy:+:vx:) 
    :tauxz:     = :mu:*(:uz:+:wx:) 
    :tauyz:     = :mu:*(:vz:+:wz:) 
    [:tx:,:ty:,:tz:] = grad(:T:)
    :kappa:     = 1.0e-3 * gbar( ring(:T:)* :rho:*:cv:/(:T: * :dt: ) )
    :cs:        = sqrt( :p: / :rho: * :gamma: )
    # Time step control routines
    :dt:        = dt.courant(:u:,:v:,:w:,:cs:)*1.0
    :dt:        = numpy.minimum(:dt:,0.2 * dt.diff(:beta:,:rho:))
    :dt:        = numpy.minimum(:dt:,0.2 * dt.diff(:mu:,:rho:))
    :dt:        = numpy.minimum(:dt:,0.2 * dt.diff(:adiff:,:rho:))
    # Add some BCs in the x-direction
    bc.const(['Yh'],['xn'],1.0)
    bc.const(['Yh'],['x1'],0.0)
    bc.const(['Yl'],['x1'],1.0)
    bc.const(['Yl'],['xn'],0.0)
    bc.extrap(['rho','Et'],['x1','xn'])
    bc.const(['u','v','w'],['x1','xn'],0.0)
    :rhoYh: = :rho:*:Yh:
    :rhoYl: = :rho:*:Yl:
    :rhou: = :rho:*:u:
    :rhov: = :rho:*:v:
    :rhow: = :rho:*:w:
    :mix:  = 4.0*:Yh:*:Yl:
    :YhYh: = :Yh:*:Yh:
    """

    # Add the EOM to the solver
    ss.EOM(textwrap.dedent(eom), parm_dict)

    # Initialize variables
    ic_vel = """
    :gamma:= 5./3.
    p0     = 1.0
    At     = ( (rho_h - rho_l)/(rho_h + rho_l) )
    # NOTE: WHY THIS PARTICULAR FRACTION OF THE RT GROWTH RATE?


    # Add interface
    # NOTE: diffuse interface?
    :Yl: = .5 * (1.0-tanh( sqrt(pi)*( meshx ) / delta ))
    :Yh: = 1.0 - :Yl:
    :p:  += p0 

    # Add oscillations near interface
    wgt = 4*:Yh:*:Yl:
    :v: *= 0.0
    :w: *= 0.0

    # gbar is 2D gaussian filter (smoothes stuff out)
    # Add random velocity (filtered white noise)
    u0  = sqrt( abs(gx*At/waveLength) ) * ranVelMag
    :u: = wgt * gbar( (random3D()-0.5)*u0 )

    # Add regular modes
    u1  = sqrt( abs(gx*At/waveLength) ) * velMag
    :u: = :u: + wgt * u1 * - cos( meshy * velMod )

    :rho:       = rho_h * :Yh: + rho_l * :Yl:

    # Solve hydro-static pressure
    :p: = getPressure(p0)

    :cv:        = :Yh:*CVh + :Yl:*CVl
    :cp:        = :Yh:*CPh + :Yl:*CPl
    :gamma:     = :cp:/:cv:
    :mw:        = 1.0 / ( :Yh: / mwH + :Yl: / mwL )
    :R:         = Runiv / :mw:
    :T:         = :p: / (:rho: * :R: )

    # Form conserved quatities
    :rhoYh: = :rho:*:Yh:
    :rhoYl: = :rho:*:Yl:
    :rhou: = :rho:*:u:
    :rhov: = :rho:*:v:
    :rhow: = :rho:*:w:
    :Et:  = :p: / (:gamma:-1.0) + 0.5*:rho:*(:u:*:u: + :v:*:v: + :w:*:w:)
    :cs:  = sqrt( :p: / :rho: * :gamma: )
    :dt: = dt.courant(:u:,:v:,:w:,:cs:)
    """

    # Velocity free/shape only interface perturbatiosn
    # NOTE: hydrostatic condition is function of y now, so need to tweak this variant
    ic_shape = """
    :gamma:= 5./3.
    p0     = 1.0
    At     = ( (rho_h - rho_l)/(rho_h + rho_l) )
    u0     = sqrt( abs(gx*At/waveLength) ) * .1  # NOTE: WHY THIS PARTICULAR FRACTION OF THE RT GROWTH RATE?

    # Perturbation: fixed single mode sinusoids
    Amp = 3.5 + 0.05*(sin(meshy*waveLength) + cos(meshz*waveLength))
    # Add interface
    :Yl: = 0.5 * (1.0 - tanh( sqrt(pi)*( meshx - Amp) / delta))
    :Yh: = 1.0 - :Yl:
    :p:  += p0 

    # Add oscillations near interface
    wgt = 4*:Yh:*:Yl:
    :v: *= 0.0
    :w: *= 0.0
    :u: *= 0.0

    :rho:       = rho_h * :Yh: + rho_l * :Yl:
    :cv:        = :Yh:*CVh + :Yl:*CVl
    :cp:        = :Yh:*CPh + :Yl:*CPl
    :gamma:     = :cp:/:cv:
    :mw:        = 1.0 / ( :Yh: / mwH + :Yl: / mwL )
    :R:         = Runiv / :mw:
    :T:         = :p: / (:rho: * :R: )

    # Form conserved quatities
    :rhoYh: = :rho:*:Yh:
    :rhoYl: = :rho:*:Yl:
    :rhou: = :rho:*:u:
    :rhov: = :rho:*:v:
    :rhow: = :rho:*:w:
    :Et:  = :p: / (:gamma:-1.0) + 0.5*:rho:*(:u:*:u: + :v:*:v: + :w:*:w:)
    :cs:  = sqrt( :p: / :rho: * :gamma: )
    :dt: = dt.courant(:u:,:v:,:w:,:cs:)
    """
    # Set the initial conditions
    # Ensure each rank has different ic to avoid periodic ics repeating across the ranks
    # ss.PyMPI.comm.rank
    if not ss.PyMPI.master:
        rank = ss.PyMPI.comm.rank
        # rank = 1
        # rank = 1
    else:
        rank = 0

    if not random_ic_seed:
        rng = numpy.random.default_rng()
        random_ic_seed = rng.integers(1000000, size=1)[0]  # Get new base seed integer
        
    print(f"Using {random_ic_seed=} as the base for seeding random number generator.  Rank is added to this in parallel.")
    
    # print(f"seed {random_ic_seed}, seed + rank: {random_ic_seed + rank}")
    numpy.random.seed(random_ic_seed + rank)
    # numpy.random.seed(random_ic_seed)  # 1234 was previous default

    # INSERT SWITCHYARD FOR ALTERNATE IC'S WHEN THEY'RE WORKING
    ss.setIC(textwrap.dedent(ic_vel), parm_dict)

    # Write a time loop
    time = 0.0

    # Start time loop
    CFL = 1.0
    dt = ss.variables["dt"].data * CFL

    # Viz/IO
    viz_freq = args.viz_freq_cycle
    dmp_freq = args.dump_freq_cycle

    tstop = 100.0
    dtmax = dt * 0.1

    outVars = ["p", "u", "v", "w", "rho", "Yh"]
    ss.write(outVars)

    def compute_mix_params(ss):
        """Helper to compute values of mix width and variance at single time state"""
        Yh = ss.var("Yh").mean(axis=[1, 2])
        x  = ss.var("meshx")[:, 0, 0]
        mixW = numpy.trapz( 6.*Yh*(1.0-Yh) , x )

        # Compute variance across y dir
        YhYh = ss.var("YhYh").mean(axis=[1, 2])
        varY = numpy.trapz( YhYh - Yh*Yh, x )

        return (mixW, varY)
            
    # Initialize time 0 data
    mwtmp, varytmp = compute_mix_params(ss)
    mixW = [mwtmp]

    varY = [varytmp]
    timeW = [0.0]

    # Setup stop criteria
    if args.tstop:
        stop_func = lambda time: time < tstop
        stop_arg = time
    elif args.stop_width_fraction:
        stop_func = lambda mixw: mixw[-1] < xwidth*args.stop_width_fraction
        stop_arg = mixW

    iter_cnt = 0
    max_iter = DEFAULTS['max_iter']
    while stop_func(stop_arg): #time < tstop:
        # Update the EOM and get next dt
        time = ss.rk4(time, dt)
        dt = ss.variables["dt"].data * CFL
        dt = min(dt, dtmax * 1.1)
        dtmax = dt * 1.0

        umax = ss.var("u").mean()
        ss.iprint("%s -- %s --- Umax: %s " % (ss.cycle, time, umax))

        mwtmp, varytmp = compute_mix_params(ss)
        mixW.append(mwtmp)

        varY.append(varytmp)
        timeW.append(time)

        if ss.cycle % dmp_freq == 0:
            ss.write(outVars)

        if not headless:
            if ss.cycle % viz_freq == 0:

                # 2D contour plots
                ss.plot.figure(1)
                ss.plot.clf()
                ss.plot.contourf("rho", 32, cmap="turbo")

                ss.plot.figure(2)
                ss.plot.clf()
                ss.plot.plot("mix", label="Mixing width")

            if ss.PyMPI.master and ss.cycle % viz_freq == 0:
                # NOTE: built in plot function expects a mesh plot.  vars vs time need
                # to fall back to raw matplotlib, and limiting it only to master mpi rank
                # NOTE: add the RT alpha growth rate ~solution for comparison?
                plt.figure(3)
                plt.plot(timeW, mixW, label='mixW')
                # plt.plot(timeW, mixWnew, label='mixWnew')
                plt.xlabel('time')
                plt.ylabel('mixing width')

                plt.figure(4)
                plt.plot(timeW, varY, label='variance')
                # plt.plot(timeW, mixWnew, label='mixWnew')
                plt.xlabel('time')
                plt.ylabel('variance')
                plt.pause(0.01)

        # Add tmp protection against infinite loops
        iter_cnt += 1
        if iter_cnt > max_iter:
            print(f"REACHED MAX ITERATION COUNT OF {max_iter}: EXITING")
            break


    # Save the mixing width curve data
    fname = problem + ".dat"
    header = f"# 'time' 'mixing width'"
    numpy.savetxt(fname, (timeW, mixW), header=header)
    print(f"Saved mixing width vs time curve to '{fname}'")  # Add more formal logger output?

    if not headless:
        plt.pause(5)

def setup_argparse():
    parser = argparse.ArgumentParser(
        prog="PyrandaRT",
        description="A configurable model of miscible Rayleigh-Taylor mixing using Pyranda",
    )

    # parser.add_argument(
    #     "-m",
    #     "--mesh-res",
    #     type=int,
    #     default=64,
    #     help="Mesh resolution to use, in terms of zones per cm, uniform in x, y, z.",
    # )

    # NOTE: maybe nice for options to have mutually exclusive arg sets, one for single density + Atwood,
    # and another for two densities and no Atwood..
    parser.add_argument(
        "-a",
        "--atwood-number",
        type=float,
        default=0.5,
        help="Atwood number to use for the simulation (density ratio).  Valid range is [0, 1). "
        "Note: At=1 sets up a fluid on top of vacuum, which is not yet supported.",
    )

    parser.add_argument(
        "-l",
        "--light-density",
        type=float,
        default=1.0,
        help="Density of the lighter fluid to use.  Heavy fluid determined via the Atwood number",
    )

    parser.add_argument(
        "-d",
        "--dim",
        type=str,
        choices={"2D", "3D"},
        default="2D",
        help="Dimensionality of the simulation to run",
    )

    parser.add_argument(
        "-n",
        "--num-points",
        type=int,
        default=64,
        help="Number of points in the y direction. x computed to ~uniform grid size.",
    )

    parser.add_argument(
        "--initial-condition",
        type=str,
        choices={"random-velocity"},
        default="random-velocity",
        help="Initial condition to run.",
    )

    parser.add_argument(
        "--random-velocity-magnitude",
        type=float,
        default=0.01,
        help="Multiplier on random velocity field added in by Pyranda",
    )

    parser.add_argument(
        "--velocity-magnitude",
        type=float,
        default=0.0,
        help="Multiplier on structured velocity field added in by Pyranda",
    )

    parser.add_argument(
        "--velocity-modes",
        type=float,
        default=1,
        help="Number of modes on structured velocity field added in by Pyranda",
    )

    parser.add_argument(
        "--velocity-thickness",
        type=float,
        default=4,
        help="Number of zones thick for initial interface and velocity field added in by Pyranda",
    )


    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        # default=1234,
        help="Seed to use in random number generator"
    )

    stop_group = parser.add_argument_group(
        title="Stop criteria",
        description="Choose one of the simulation stopping conditions (Default=tstop)"
    )
    sg_exclusive = stop_group.add_mutually_exclusive_group()
    sg_exclusive.add_argument(
        '-t',
        '--tstop',
        default=None,
        type=float,
        help=f"Simulation time to run to in units of... Default: {DEFAULTS['tstop']}"
    )
    sg_exclusive.add_argument(
        '--stop-width-fraction',
        default=None,
        type=float,
        help="Specify stopping critiera to be when mean mixing width reaches this "
        f"fraction of the x-domain width.  Range (0, 1]. Default: {DEFAULTS['stop-width-fraction']}."
    )
    # add flag for dtmax fraction?
    
    # Output controls
    parser.add_argument(
        '--viz-freq-cycle',
        type=int,
        default=20,
        help="Cycle intervals at which to dump scalar curve data."
    )

    parser.add_argument(
        '--dump-freq-cycle',
        type=int,
        default=200,
        help="Cycle intervals at which to dump mesh structured viz data (visit)."
    )

    parser.add_argument(
        '--headless',
        action='store_true',
        help="Turn on headless mode to hide interactive plots"
    )
    
    # Potential other interesting args:
    # Problem name -> this controls output dir?
    # domain size (x, y, z), npts x, y, z
    # scaling factor on initial growth rate/velocity -> wavelength used here, but not for shape??
    # Dump pngs from matplotlib for easier movie automation?  -> does slow down sim a bit; may
    #      need to optimize the pyrandaplot funcs to append data instead of blowing away each frame
    return parser


def main():
    parser = setup_argparse()

    args = parser.parse_args()

    try:
        run_sim(args)
    except Exception:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
