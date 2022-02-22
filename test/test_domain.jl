using FourierCore.Domain

@testset "Domain Construction" begin
    @test Circle() isa Circle
    @test Circle(0) isa Circle
    @test Circle(0) × Circle(0) isa Torus
    @test Circle(0)^2  isa Torus
    @test Circle(0)^3 isa  Torus
    T¹ = S¹(2π) × S¹(2π)
    @test(T¹ × S¹(1π) isa Torus) # note π is an irrational type
end

@testset "Correct Values in Correct Locations" begin
    S = Circle(1.0, 2.0)
    @test S.a == 1.0
    @test S.b == 2.0
    T¹ = S¹(2, 3) × S¹(5.0, 7.0)
    @test T¹[1].a == 2
    @test T¹[1].b == 3
    @test T¹[2].a == 5.0
    @test T¹[2].b == 7.0
end