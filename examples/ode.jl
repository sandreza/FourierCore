

function solve!(q, timeend, solver;
                after_step::Function = (x...) -> nothing,
                after_stage::Function = (x...) -> nothing,
                adjust_final = true)
  finalstep = false
  step = 0
  while true
    step += 1
    time = solver.time
    if time + solver.dt >= timeend
      adjust_final && (solver.dt = timeend - time)
      finalstep = true
    end
    dostep!(q, solver)
    after_step(step, time, q)
    finalstep && break
  end
end

mutable struct LSRK{FT, AT, NS, RHS}
  dt::FT
  time::FT
  rhs!::RHS
  dq::AT
  rka::NTuple{NS, FT}
  rkb::NTuple{NS, FT}
  rkc::NTuple{NS, FT}

  function LSRK(rhs!, rka, rkb, rkc, q, dt, t0)
      FT = eltype(eltype(q))
      dq = copy(q)
      fill!(dq, zero(eltype(q)))
      AT = typeof(dq)
      RHS = typeof(rhs!)
      new{FT, AT, length(rka), RHS}(FT(dt), FT(t0), rhs!, dq, rka, rkb, rkc)
  end
end

function dostep!(q, lsrk::LSRK)
  (; rhs!, dq, rka, rkb, rkc, dt, time) = lsrk
  for stage = 1:length(rka)
    stagetime = time + rkc[stage] * dt
    dq .*= rka[stage]
    rhs!(dq, q, stagetime)
    @. q += rkb[stage] * dt * dq
  end
  lsrk.time += dt
end

function LSRK144(rhs!, q, dt; t0=0)
    rka, rkb, rkc = coefficients_lsrk144()
    LSRK(rhs!, rka, rkb, rkc, q, dt, t0)
end

function coefficients_lsrk144()
    rka = (
        0,
        -0.7188012108672410,
        -0.7785331173421570,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.7483334182761610,
        -0.7032861106563359,
        0.0013917096117681,
        -0.0932075369637460,
        -0.9514200470875948,
        -7.1151571693922548,
    )

    rkb = (
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.3326293790646110,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628,
    )

    rkc = (
        0,
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.2476149531070420,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976,
    )

    rka, rkb, rkc
end
