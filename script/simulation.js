/* 
 * simulation.js
 * Created by Jorge Gonzalez, January, 10, 2019.
 * Released under MIT License - see LICENSE file for details.
 */

"use strict";

var WALL_CATEGORY = 1;
var CREATURE_CATEGORY = 2;
var FOOD_CATEGORY = 3;

var FOOD_PROBABILITY = 0.995;
var MINIMUM_FOOD = 10;

var WORLD_WIDTH = 1000;
var WORLD_HEIGHT = 500;
var WORLD_OFFSET = cp.v(150,100);
var worldPoints = [
 [WORLD_OFFSET.x, WORLD_OFFSET.y], 
 [WORLD_OFFSET.x + WORLD_WIDTH, WORLD_OFFSET.y], 
 [WORLD_OFFSET.x + WORLD_WIDTH, WORLD_OFFSET.y + WORLD_HEIGHT], 
 [WORLD_OFFSET.x, WORLD_OFFSET.y + WORLD_HEIGHT]
]; 

var now = 1;

class Simulation {

    constructor(renderer) {
        this.space = new cp.Space();
        this.renderer = renderer;

        //Configure space
        var space = this.space;
        space.iterations = 30;
        space.gravity = cp.v(0, 0);
        space.damping = 0.1;
        space.sleepTimeThreshold = 0.5;
        space.collisionSlop = 0.5;
        space.addCollisionHandler(CREATURE_CATEGORY, WALL_CATEGORY, this.handleCreatureWallCollision, null, null, null);

        this.createWorld(worldPoints);
        this.creaturesInitPosition = cp.v(300,300);

        this.generationCount = 0;
        this.generationBest = 0;
        this.generationBestPrevious = 0;  
        this.generationStartTime = window.now;

        this.bestLapTime = Number.MAX_SAFE_INTEGER;

        this.eyeTracingCheckBox = document.getElementById('eyeTracing');
        this.mutationProbabilityEdit = document.getElementById('mutProb');
        this.mutationChangeEdit = document.getElementById('mutChange');
        this.numCreaturesEdit = document.getElementById('numCreatures');
        this.deadOnCollisionCheckBox = document.getElementById('deadOnCollision');
        this.deadByOldCheckBox = document.getElementById('deadByOld');
        this.useTanhEdit = document.getElementById('useTanh');
        this.generationTimeToLiveEdit = document.getElementById('genTTL');

        this.generationTimeToLive = this.generationTimeToLiveEdit.value;

        // Creatures
        this.creatures = [];
        for(let i = 0; i < this.numCreaturesEdit.value; ++i) {
        //for(let i = 0; i < 2; ++i) {
            this.creatures.push(new Creature(this.space, this.randomPosition, CREATURE_CATEGORY, CreatureType.new));
        }
        
        var self = this;
        document.addEventListener('keydown', function(event) {
            var creature = self.creatures[0];
            var rot = creature.shape.body.rot;
            if(event.keyCode == 87)  // w
            {
                creature.shape.body.applyImpulse( vmult(rot, 10), cp.v(0, 0));
            }
            if(event.keyCode == 83)  // s
            {
                creature.shape.body.applyImpulse( vmult(rot, -10), cp.v(0, 0));
            }
            if(event.keyCode == 65)  // a
            {
                creature.shape.body.applyImpulse( cp.v(-10, 0), cp.v(0, 1));
                creature.shape.body.applyImpulse( cp.v(10, 0), cp.v(0, -1));
            }
            if(event.keyCode == 68)  // d
            {
                creature.shape.body.applyImpulse( cp.v(10, 0), cp.v(0, 1));
                creature.shape.body.applyImpulse( cp.v(-10, 0), cp.v(0, -1));
            }
            if(event.keyCode == 80)  // p
            {
                console.log(self.creatures[0].fitness(self.checkpoints));
            }
        });
    }


    createWorld(worldPoints) {
        var self = this;
        let addWall = function(p0, p1) {
            let wall = self.space.addShape(new cp.SegmentShape(self.space.staticBody, cp.v(p0[0], p0[1]), cp.v(p1[0], p1[1]), 0));
            wall.setElasticity(1);
            wall.setFriction(1);
            wall.setCollisionType(WALL_CATEGORY);
            wall.group = WALL_CATEGORY;
            self.static_walls.push(wall); 
        }

        this.static_walls = [];

        for (let i = 0; i < worldPoints.length; ++i) {
            let p0 = worldPoints[i];
            let p1 = worldPoints[(i + 1) % worldPoints.length];

            addWall(p0, p1);        
        }
    }

    handleCreatureWallCollision(arb, space) {
        if(self.sim.deadOnCollisionCheckBox.checked) {
            var c = arb.a.collision_type == CREATURE_CATEGORY ? arb.a : arb.b;
            c.creature.alive = false;
        }
        //console.log("Death by collision");
        return true;
    }

    run() {
        this.running = true;
        var self = this;
        var lastTime = 0;
        var step = function (time) {

            self.update(time - lastTime);
            self.draw();

            lastTime = time;

            if (self.running) {
                window.requestAnimationFrame(step);
            }
        };
        step(0);
    }

    randomPosition() {
        let border = 30;
        var posx = WORLD_OFFSET.x + border + (WORLD_WIDTH - 2 * border) * Math.random();
        var posy = WORLD_OFFSET.y + border + (WORLD_HEIGHT - 2 * border) * Math.random();
        return new cp.v(posx, posy);
    }

    update(dt) { 
        let now = self.now;
        this.space.step(1/60);
        self.now += 1000/60;

        let mutationProbability = this.mutationProbabilityEdit.value;
        let mutationChange = this.mutationChangeEdit.value;
        let numCreatures = this.numCreaturesEdit.value;
    
        for(let i = 0; i < this.creatures.length; ++i) {
            this.creatures[i].update();
            this.creatures[i].showEyeTracing = this.eyeTracingCheckBox.checked;
        }

        if(!this.deadByOldCheckBox.checked)
            this.generationStartTime = now;

        if(now - this.generationStartTime > this.generationTimeToLive * 1000) {
            this.creatures.forEach(c => {
                if(c.alive) {
                    c.alive = false;
                    console.log("Death by old age");
                }
            });            
        }
    
        if(this.creatures.filter(c => c.alive).length <= 1) {
    
            for(let i = 0; i < this.creatures.length; ++i) {
                this.space.removeShape(this.creatures[i].shape);
                this.space.removeBody(this.creatures[i].body);
            }
            
            this.creatures.sort((c1, c2) => c2.fitness() - c1.fitness());
            let numSurvivors = Math.floor(numCreatures / 3);
            var survivors = this.creatures.slice(0, numSurvivors);
            var first = survivors[0];
            var second = survivors[1];
            this.creatures = [];
            
            this.generationBestPrevious = this.generationBest;
            this.generationBest = first.fitness();
            console.log("Generation best: " + this.generationBest);
    
            // crosses
            for(let i = 0; i < survivors.length - 1; ++i) {
                for(let j = i + 1; j < survivors.length; ++j) {
                    let cc = survivors[i].clone();
                    cc.crossover(survivors[j]);
                    cc.creatureType = CreatureType.cross;
                    this.creatures.push(cc.mutate(0.02, 0.02));
                }
            }
    
            // clone first
            let fclone = first.clone();
            fclone.creatureType = CreatureType.first;
            this.creatures.push(fclone)

            // fine mutations
            let numFimeMutations = numCreatures >= 10 ? 2 : 1;
            for(let i = 0; i < numFimeMutations; ++i) {
                let mutationOfFirst = first.clone().mutate(0.85, 0.05);
                mutationOfFirst.creatureType = CreatureType.fineMutation;
                this.creatures.push(mutationOfFirst);
            }
    
            // coarse mutations
            let creatures_len = this.creatures.length;
            for(let i = 0; i < numCreatures - creatures_len; ++i) {
                let mutationOfFirst = first.clone().mutate(mutationProbability, mutationChange);
                mutationOfFirst.creatureType = CreatureType.mutation;
                this.creatures.push(mutationOfFirst);
            }

            if(this.generationCount < 5) {
                this.creatures.push(new Creature(this.space, this.randomPosition, CREATURE_CATEGORY, CreatureType.new));
                this.creatures.push(new Creature(this.space, this.randomPosition, CREATURE_CATEGORY, CreatureType.new));
            }
    
            ++this.generationCount;
            this.generationStartTime = now;
            this.generationTimeToLive = this.generationTimeToLiveEdit.value;
        }
    }

    draw() {
        this.renderer.clear();

        for(let i = 0; i < this.creatures.length; ++i)        
            this.renderer.draw(this.creatures[i]);             

        for(let i = 0; i < this.static_walls.length; ++i)        
            this.renderer.draw(this.static_walls[i]);            

        let text = "Generation: " + this.generationCount;
        text += "   Last: " + this.generationBest.toFixed(2);
        text += "   Previous: " + this.generationBestPrevious.toFixed(2);
        text += "   Remaining gen time: " + (this.deadByOldCheckBox.checked ? (Math.floor((this.generationTimeToLive * 1000 - (self.now - this.generationStartTime)) / 1000) + "s") : "--");
        text += "   Best lap: " + ((this.bestLapTime != Number.MAX_SAFE_INTEGER) ? ((this.bestLapTime/1000).toFixed(2) + "s") : "N/A");        
        
        this.renderer.printInfo(text);
    }
};




